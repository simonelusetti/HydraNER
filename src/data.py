import os
import tarfile
import zipfile
import logging
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
import torch
import numpy as np
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel
from dora import to_absolute_path
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def _bio_labels(*labels: str) -> list[str]:
    names = ["O"]
    for label in labels:
        names.append(f"B-{label}")
        names.append(f"I-{label}")
    return names


NER_LABEL_NAMES = {
    "wikiann": ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
    "conll2003": ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],
    "wnut": [
        "O",
        "B-corporation",
        "I-corporation",
        "B-creative-work",
        "I-creative-work",
        "B-group",
        "I-group",
        "B-location",
        "I-location",
        "B-person",
        "I-person",
        "B-product",
        "I-product",
    ],
    "ontonotes": _bio_labels(
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ),
    "bc2gm": ["O", "B-GENE", "I-GENE"],
}

WNUT_LABEL_TO_ID = {label: idx for idx, label in enumerate(NER_LABEL_NAMES["wnut"])}
ONTONOTES_LABEL_TO_ID = {label: idx for idx, label in enumerate(NER_LABEL_NAMES["ontonotes"])}
BC2GM_LABEL_TO_ID = {label: idx for idx, label in enumerate(NER_LABEL_NAMES["bc2gm"])}


# ---------- Helpers ----------

def _is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip() == "1" or os.environ.get("HF_DATASETS_OFFLINE", "").strip() == "1"


def _sanitize_fragment(fragment: str) -> str:
    return fragment.replace("/", "-")


def _dataset_cache_filename(name, split, subset, cnn_field=None, dataset_config=None):
    parts = [name]
    if dataset_config:
        parts.append(_sanitize_fragment(dataset_config))
    if cnn_field:
        parts.append(cnn_field)
    parts.append(split)
    if subset is not None and subset != 1.0:
        parts.append(str(subset))
    return "_".join(parts) + ".pt"


def _raw_data_dir() -> Path:
    path = Path(to_absolute_path("./.dataset_downloads"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _freeze_encoder(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def _encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels=None):
    keep_labels = keep_labels or []

    def _tokenize_and_encode(x):
        # If we have ner_tags and tokens, align ner_tags to subword tokens
        has_ner = "ner_tags" in x and "tokens" in x
        if has_ner:
            enc = tok(x["tokens"], truncation=True, max_length=max_length, is_split_into_words=True)
        else:
            enc = tok(text_fn(x), truncation=True, max_length=max_length)

        device = next(encoder.parameters()).device
        inputs = {
            "input_ids": torch.tensor(enc["input_ids"], device=device).unsqueeze(0),
            "attention_mask": torch.tensor(enc["attention_mask"], device=device).unsqueeze(0),
        }
        with torch.no_grad():
            out = encoder(**inputs, output_attentions=False, return_dict=True)

            out_dict = {
                "input_ids": np.asarray(enc["input_ids"], dtype=np.int64),
                "attention_mask": np.asarray(enc["attention_mask"], dtype=np.int64),
                "embeddings": out.last_hidden_state.squeeze(0).detach().cpu().to(torch.float32).numpy(),
            }
            # Align ner_tags to subword tokens if present
            if has_ner:
                word_ids = enc.word_ids()
                ner_tags = x["ner_tags"]
                aligned_ner_tags = []
                for word_id in word_ids:
                    if word_id is None:
                        aligned_ner_tags.append(0)  # or -100 for ignore, but 0 = O
                    else:
                        aligned_ner_tags.append(ner_tags[word_id])
                out_dict["ner_tags"] = np.asarray(aligned_ner_tags, dtype=np.int64)
                for k in keep_labels:
                    if k not in ["ner_tags", "tokens"]:
                        out_dict[k] = x[k]
                # Optionally keep tokens for debugging
                out_dict["tokens"] = x["tokens"]
            else:
                for k in keep_labels:
                    out_dict[k] = x[k]
            return out_dict

    return ds.map(_tokenize_and_encode, remove_columns=ds.column_names, batched=False)




def _normalize_ontonotes_split(split: str) -> str:
    normalized = split.lower().strip()
    mapping = {"train": "train", "training": "train", "validation": "development", "dev": "development", "val": "development", "test": "test"}
    if normalized not in mapping:
        raise ValueError(f"Unsupported OntoNotes split '{split}'.")
    return mapping[normalized]


def _ontonotes_data_root(version: str) -> Path:
    base = _raw_data_dir() / "conll-2012" / version
    data_dir = base / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"OntoNotes data directory '{data_dir}' not found. "
            f"Download and extract the official archive into '{base}'."
        )
    return data_dir


def _ontonotes_split_dir(version: str, language: str, split: str) -> Path:
    normalized = _normalize_ontonotes_split(split)
    subdir = {
        "train": "train/data",
        "development": "development/data",
        "test": "test/data",
    }[normalized]
    path = _ontonotes_data_root(version) / subdir / language / "annotations"
    if not path.exists():
        raise FileNotFoundError(f"OntoNotes annotations directory '{path}' missing for split '{split}'.")
    return path


def _ontonotes_tag_to_label(raw_tag: str, current: str | None) -> tuple[str, str | None]:
    tag = raw_tag.strip()
    if "|" in tag:
        tag = tag.split("|")[0]
    if tag == "*":
        if current is None:
            return "O", current
        return f"I-{current}", current
    if tag.startswith("("):
        entity = tag[1:].strip("*)")
        label = f"B-{entity}"
        if not tag.endswith(")"):
            current = entity
        else:
            current = None
        return label, current
    if tag.endswith(")"):
        if current is None:
            return "O", None
        label = f"I-{current}"
        return label, None
    raise ValueError(f"Unrecognized OntoNotes tag '{raw_tag}'.")


def _parse_ontonotes_file(path: Path, label_to_id: dict[str, int]) -> list[dict]:
    sentences = []
    tokens = []
    labels = []
    current_entity = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                if tokens:
                    sentences.append({
                        "tokens": tokens,
                        "ner_tags": [label_to_id[label] for label in labels],
                    })
                    tokens = []
                    labels = []
                current_entity = None
                continue
            if stripped.startswith("#"):
                current_entity = None
                continue
            parts = stripped.split()
            if len(parts) < 11:
                continue
            token = parts[3]
            raw_tag = parts[10]
            label, current_entity = _ontonotes_tag_to_label(raw_tag, current_entity)
            if label not in label_to_id:
                raise ValueError(f"Encountered OntoNotes label '{label}' not in label map.")
            tokens.append(token)
            labels.append(label)
    if tokens:
        sentences.append({
            "tokens": tokens,
            "ner_tags": [label_to_id[label] for label in labels],
        })
    return sentences


def _load_ontonotes_dataset(split: str, config_name: str):
    normalized_split = _normalize_ontonotes_split(split)
    try:
        ds = load_dataset("ontonotes5", config_name, split=normalized_split)
    except Exception as err:  # pragma: no cover - requires HF download
        logger.warning("Falling back to local OntoNotes parser due to: %s", err)
        return _load_ontonotes_dataset_local(split, config_name)

    def _find_column(candidates):
        for name in candidates:
            if name in ds.column_names:
                return name
        return None

    tokens_field = _find_column(("tokens", "words"))
    labels_field = _find_column(("ner_tags", "ner", "named_entities"))
    if tokens_field is None or labels_field is None:
        raise ValueError(
            "Unexpected OntoNotes dataset structure; expected token and ner tag columns."
        )
    if tokens_field != "tokens":
        ds = ds.rename_column(tokens_field, "tokens")
    if labels_field != "ner_tags":
        ds = ds.rename_column(labels_field, "ner_tags")

    feature = ds.features.get("ner_tags")
    hf_label_names = None
    if feature is not None:
        seq_feature = getattr(feature, "feature", None)
        hf_label_names = getattr(seq_feature, "names", None)

    def _convert_label(value):
        if isinstance(value, int):
            if hf_label_names and 0 <= value < len(hf_label_names):
                label_name = hf_label_names[value]
            else:
                return int(value)
        else:
            label_name = str(value)
        label_name = label_name.strip().upper()
        try:
            return ONTONOTES_LABEL_TO_ID[label_name]
        except KeyError as err:
            raise ValueError(f"Unknown OntoNotes label '{label_name}'.") from err

    def _standardize(example):
        example["ner_tags"] = [_convert_label(label) for label in example["ner_tags"]]
        return example

    return ds.map(_standardize, desc="Standardizing OntoNotes labels")


def _load_ontonotes_dataset_local(split: str, config_name: str):
    try:
        language, version = config_name.split("_", 1)
    except ValueError as err:
        raise ValueError(f"Invalid OntoNotes config '{config_name}'. Expected format '<language>_<version>'.") from err
    if language != "english":
        raise ValueError(f"Only English OntoNotes is supported (got language='{language}').")
    annotations_dir = _ontonotes_split_dir(version, language, split)
    patterns = ["*.gold_conll", "*.v4_gold_conll", "*.v9_gold_conll", "*.v12_gold_conll"]
    files = []
    for pattern in patterns:
        files.extend(annotations_dir.rglob(pattern))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No OntoNotes files found under {annotations_dir}")
    label_map = ONTONOTES_LABEL_TO_ID
    examples = []
    for file_path in files:
        examples.extend(_parse_ontonotes_file(file_path, label_map))
    return Dataset.from_list(examples)


BC2GM_SPLIT_NAMES = {
    "train": ["gene.train"],
    "validation": ["gene.dev", "gene.devel", "gene.eval"],
    "test": ["gene.test"],
}


def _extract_archive(archive_path: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)
    else:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(destination)


def _locate_bc2gm_root() -> Path:
    raw_dir = _raw_data_dir()
    def _probe_directories():
        for candidate in raw_dir.iterdir():
            if not candidate.is_dir():
                continue
            name = candidate.name.lower()
            if "bc2" not in name and "gene" not in name:
                continue
            for split_targets in BC2GM_SPLIT_NAMES.values():
                for target in split_targets:
                    if any(candidate.rglob(f"*{target}*")):
                        return candidate
        return None
    existing = _probe_directories()
    if existing is not None:
        return existing
    archives = sorted(
        list(raw_dir.glob("bc2gm*.tar*")) + list(raw_dir.glob("bc2gm*.zip")),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if not archives:
        raise FileNotFoundError(
            f"BC2GM archive not found under {raw_dir}. "
            "Download the official BioCreative II Gene Mention corpus (e.g., bc2gm_corpus.tar.gz) into this directory."
        )
    for archive in archives:
        _extract_archive(archive, raw_dir)
    existing = _probe_directories()
    if existing is None:
        raise FileNotFoundError(
            f"Unable to locate BC2GM data under {raw_dir} even after extracting archives. "
            "Ensure the archive contains files like GENE.train/GENE.dev/GENE.test."
        )
    return existing


def _find_bc2gm_split_file(root: Path, split: str) -> Path:
    targets = BC2GM_SPLIT_NAMES.get(split)
    if targets is None:
        raise ValueError(f"Unsupported BC2GM split '{split}'. Use 'train', 'validation', or 'test'.")
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        name = file_path.name.lower()
        for target in targets:
            if name == target or name.startswith(f"{target}."):
                return file_path
    raise FileNotFoundError(
        f"No BC2GM file found for split '{split}'. Expected one of: {targets}. "
        f"Ensure the archive contains the standard BioCreative II files."
    )


def _parse_bc2gm_file(path: Path):
    examples = []
    tokens = []
    labels = []
    label_map = BC2GM_LABEL_TO_ID
    doc_id = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    examples.append(
                        {
                            "id": str(doc_id),
                            "tokens": tokens,
                            "ner_tags": [label_map[label] for label in labels],
                        }
                    )
                    tokens = []
                    labels = []
                    doc_id += 1
                continue
            if stripped.startswith("#"):
                if tokens:
                    examples.append(
                        {
                            "id": str(doc_id),
                            "tokens": tokens,
                            "ner_tags": [label_map[label] for label in labels],
                        }
                    )
                    tokens = []
                    labels = []
                    doc_id += 1
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            token = parts[0]
            tag = parts[-1]
            if tag not in label_map:
                raise ValueError(f"Unknown BC2GM label '{tag}' encountered in {path}.")
            tokens.append(token)
            labels.append(tag)
    if tokens:
        examples.append(
            {
                "id": str(doc_id),
                "tokens": tokens,
                "ner_tags": [label_map[label] for label in labels],
            }
        )
    return examples


def _load_bc2gm_dataset(split: str):
    root = _locate_bc2gm_root()
    file_path = _find_bc2gm_split_file(root, split)
    examples = _parse_bc2gm_file(file_path)
    return Dataset.from_list(examples)


def build_dataset(
    name,
    split,
    tokenizer_name,
    max_length,
    subset=None,
    shuffle=False,
    cnn_field=None,
    dataset_config=None,
    raw_dataset_root=None,
):
    """
    Generic dataset builder for CNN, WikiANN, CoNLL, WNUT, OntoNotes, BC2GM, and FrameNet.
    """
    # pick dataset + text extraction strategy
    raw_split_path = None
    if raw_dataset_root is not None:
        raw_split_path = Path(raw_dataset_root) / split
        if not raw_split_path.exists():
            raise FileNotFoundError(f"Raw dataset split not found at {raw_split_path}")

    if name == "cnn":
        if raw_split_path is not None:
            ds = load_from_disk(str(raw_split_path))
        else:
            ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        if cnn_field is None: cnn_field = "highlights"
        text_fn = lambda x: x[cnn_field]
        keep_labels = []
    elif name == "wikiann":
        if raw_split_path is not None:
            ds = load_from_disk(str(raw_split_path))
        else:
            ds = load_dataset("wikiann", "en", split=split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "conll2003":
        if raw_split_path is not None:
            ds = load_from_disk(str(raw_split_path))
        else:
            ds = load_dataset("conll2003", revision="refs/convert/parquet", split=split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "wnut":
        if raw_split_path is None:
            raise RuntimeError(
                "WNUT requires pre-downloaded raw splits. "
                "Run `python ../tools/datasets/download_dataset.py --dataset wnut --output data/raw/wnut` first."
            )
        ds = load_from_disk(str(raw_split_path))
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "ontonotes":
        config_name = dataset_config or "english_v4"
        if raw_split_path is not None:
            ds = load_from_disk(str(raw_split_path))
        else:
            ds = _load_ontonotes_dataset(split, config_name)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "bc2gm":
        try:
            ds = load_dataset("spyysalo/bc2gm_corpus", split=split)
        except Exception as err:
            logger.warning("Falling back to local BC2GM parser due to: %s", err)
            ds = _load_bc2gm_dataset(split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "framenet":
        if raw_split_path is None:
            raise RuntimeError(
                "FrameNet requires pre-downloaded raw splits. "
                "Run `python ../tools/datasets/download_dataset.py --dataset framenet --output data/raw` first."
            )
        ds = load_from_disk(str(raw_split_path))
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["tokens", "frame_elements", "frame_name", "lexical_units", "lemmas", "pos_tags"]
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    if shuffle:
        ds = ds.shuffle(seed=42)

    if subset is not None:
        if subset <= 1.0:
            subset = int(len(ds) * subset)
        ds = ds.select(range(subset))

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = _freeze_encoder(AutoModel.from_pretrained(tokenizer_name))
    ds = _encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels)
    return ds, tok

def initialize_dataloaders(cfg, logger):
    train_cfg = cfg.data.train
    eval_cfg = cfg.data.eval
    dev_cfg = cfg.data.dev

    train_cnn_field = train_cfg.cnn_field if hasattr(train_cfg, "cnn_field") else None
    train_ds, _ = get_dataset(
        name=train_cfg.dataset,
        subset=train_cfg.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=train_cfg.shuffle,
        dataset_config=train_cfg.config,
        cnn_field=train_cnn_field,
    )
    eval_shuffle = bool(eval_cfg.shuffle)
    if eval_shuffle:
        logger.warning("Disabling shuffle for expert evaluation loader to preserve ordering.")
        eval_shuffle = False

    eval_split = eval_cfg.split if hasattr(eval_cfg, "split") else "validation"
    eval_cnn_field = eval_cfg.cnn_field if hasattr(eval_cfg, "cnn_field") else None
    eval_ds, _ = get_dataset(
        split=eval_split,
        name=eval_cfg.dataset,
        subset=eval_cfg.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=eval_shuffle,
        dataset_config=eval_cfg.config,
        cnn_field=eval_cnn_field,
    )

    dev_dl = None
    if dev_cfg and dev_cfg.dataset:
        dev_cnn_field = dev_cfg.cnn_field if hasattr(dev_cfg, "cnn_field") else None
        dev_split = dev_cfg.split if hasattr(dev_cfg, "split") else "test"
        dev_ds, _ = get_dataset(
            split=dev_split,
            name=dev_cfg.dataset,
            subset=dev_cfg.subset,
            rebuild=cfg.data.rebuild_ds,
            shuffle=bool(dev_cfg.shuffle),
            dataset_config=dev_cfg.config,
            cnn_field=dev_cnn_field,
        )
        dev_dl = DataLoader(
            dev_ds,
            batch_size=dev_cfg.batch_size,
            collate_fn=collate,
            num_workers=dev_cfg.num_workers,
            pin_memory=(cfg.runtime.device == "cuda"),
            persistent_workers=(dev_cfg.num_workers > 0),
            shuffle=bool(dev_cfg.shuffle),
        )

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        collate_fn=collate,
        num_workers=train_cfg.num_workers,
        pin_memory=(cfg.runtime.device == "cuda"),
        persistent_workers=(train_cfg.num_workers > 0),
        shuffle=train_cfg.shuffle,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=eval_cfg.batch_size,
        collate_fn=collate,
        num_workers=eval_cfg.num_workers,
        pin_memory=(cfg.runtime.device == "cuda"),
        persistent_workers=(eval_cfg.num_workers > 0),
        shuffle=eval_shuffle,
    )

    train_label_names = NER_LABEL_NAMES.get(cfg.data.train.dataset)
    if train_label_names:
        setattr(train_dl, "label_names", train_label_names)
    eval_label_names = NER_LABEL_NAMES.get(eval_cfg.dataset)
    if eval_label_names:
        setattr(eval_dl, "label_names", eval_label_names)
    if dev_dl is not None:
        dev_label_names = NER_LABEL_NAMES.get(dev_cfg.dataset)
        if dev_label_names:
            setattr(dev_dl, "label_names", dev_label_names)
    return train_dl, eval_dl, dev_dl

# ---------- Collate ----------

from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    # assume batch is a list of dicts
    def _as_tensor(value, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype)
        return torch.tensor(value, dtype=dtype)

    input_ids = [_as_tensor(x["input_ids"], torch.long) for x in batch]
    attention_masks = [_as_tensor(x["attention_mask"], torch.long) for x in batch]

    has_ner = "ner_tags" in batch[0]
    if has_ner:
        ner_tags = [_as_tensor(x["ner_tags"], torch.long) for x in batch]

    # pad to longest sequence in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    if has_ner:
        ner_tags = pad_sequence(ner_tags, batch_first=True, padding_value=-100)  # -100 is common ignore_index

    batch_out = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
    }

    if has_ner:
        batch_out["ner_tags"] = ner_tags

    # add precomputed embeddings if your dataset already has them
    if "embeddings" in batch[0]:
        embeddings = [_as_tensor(x["embeddings"], torch.float) for x in batch]
        embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        batch_out["embeddings"] = embeddings

    return batch_out


# ---------- Loader ----------

def get_dataset(tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                name="cnn", split="train", max_length=256, dataset_config=None,
                cnn_field=None, subset=None, rebuild=False, shuffle=False):
    
    filename = _dataset_cache_filename(
        name,
        split,
        subset,
        cnn_field=cnn_field,
        dataset_config=dataset_config,
    )
    path = Path(to_absolute_path(f"./data/{filename}"))
    apply_subset_after_load = False
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    if rebuild:
        raise RuntimeError(
            "Dataset rebuilds are handled by ../tools/datasets/build_dataset.py. Run it before launching training."
        )

    if not os.path.exists(path):
        if subset is not None and subset != 1.0:
            fallback_filename = _dataset_cache_filename(
                name,
                split,
                None,
                cnn_field=cnn_field,
                dataset_config=dataset_config,
            )
            fallback_path = Path(to_absolute_path(f"./data/{fallback_filename}"))
            if fallback_path.exists():
                logger.warning(
                    "Subset cache %s not found; using full cache %s and selecting a subset in-memory.",
                    path,
                    fallback_path,
                )
                path = fallback_path
                apply_subset_after_load = True
            else:
                raise FileNotFoundError(
                    f"Dataset cache {path} not found. "
                    f"Run `../tools/datasets/build_dataset.py --dataset {name} --splits {split}` to materialise it."
                )
        else:
            raise FileNotFoundError(
                f"Dataset cache {path} not found. Run `../tools/datasets/build_dataset.py --dataset {name} --splits {split}` to materialise it."
            )

    logger.info(f"Loading cached dataset from {path}")
    try:
        ds = load_from_disk(path)
    except (FileNotFoundError, ValueError) as err:
        raise RuntimeError(
            "Dataset cache is unreadable. Rebuild it with ../tools/datasets/build_dataset.py."
        ) from err

    if apply_subset_after_load:
        target_subset = subset
        if target_subset <= 1.0:
            target_subset = int(len(ds) * target_subset)
        ds = ds.select(range(target_subset))
        logger.info("Materialised subset of %s examples from %s in-memory.", target_subset, path)

    if shuffle:
        ds = ds.shuffle(seed=42)

    return ds, tok
