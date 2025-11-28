from __future__ import annotations

import tarfile, zipfile, os, torch, logging
import numpy as np

from pathlib import Path
from datasets import Dataset, load_dataset
from dora import to_absolute_path

logger = logging.getLogger(__name__)

DATASETS_WITH_CONFIG = {
    "wikiann": "en",
    "ontonotes": "english_v4",
}


def sanitize_fragment(fragment: str) -> str:
    return fragment.replace("/", "-")


def normalize_dataset_config(name: str, dataset_config: str | None) -> str | None:
    """Only keep dataset_config for datasets that support it; supply defaults when needed."""
    default = DATASETS_WITH_CONFIG.get(name)
    if default is None:
        return None
    if dataset_config is None:
        return default
    return dataset_config


def dataset_cache_filename(
    name: str,
    split: str,
    subset,
    *,
    cnn_field: str | None = None,
    dataset_config: str | None = None,
) -> str:
    parts = [name]
    if dataset_config:
        parts.append(sanitize_fragment(dataset_config))
    if cnn_field:
        parts.append(cnn_field)
    parts.append(split)
    if subset is not None and subset != 1.0:
        parts.append(str(subset))
    return "_".join(parts) + ".pt"


def resolve_dataset(
    name: str,
    split: str,
    dataset_config: str | None,
    raw_dataset_root: str | Path | None,
    cnn_field: str | None,
    logger=None,
):
    """
    Centralised dataset resolution: returns (hf_dataset, text_fn, keep_labels).
    """
    from datasets import load_dataset, load_from_disk

    raw_split_path = None
    if raw_dataset_root is not None:
        raw_split_path = Path(raw_dataset_root) / split
        if not raw_split_path.exists():
            raise FileNotFoundError(f"Raw dataset split not found at {raw_split_path}")

    if name in {"cnn_highlights", "cnn"}:
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("cnn_dailymail", "3.0.0", split=split)
        if cnn_field is None:
            cnn_field = "highlights"
        text_fn = lambda x: x[cnn_field]
        keep_labels = []
    elif name == "wikiann":
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("wikiann", "en", split=split)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "conll2003":
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else load_dataset("conll2003", revision="refs/convert/parquet", split=split)
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
        ds = load_from_disk(str(raw_split_path)) if raw_split_path is not None else _load_ontonotes_dataset(split, config_name)
        text_fn = lambda x: " ".join(x["tokens"])
        keep_labels = ["ner_tags", "tokens"]
    elif name == "bc2gm":
        try:
            ds = load_dataset("spyysalo/bc2gm_corpus", split=split)
        except Exception as err:
            if logger is not None:
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

    return ds, text_fn, keep_labels


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

def _is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip() == "1" or os.environ.get("HF_DATASETS_OFFLINE", "").strip() == "1"


def _raw_data_dir() -> Path:
    path = Path(to_absolute_path("./.dataset_downloads"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels=None):
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