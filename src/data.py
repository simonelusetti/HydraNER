import os, torch, logging
import numpy as np

from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from dora import to_absolute_path
from torch.utils.data import DataLoader

from .datasets import (
    normalize_dataset_config,
    dataset_cache_filename,
    resolve_dataset,
    encode_examples,
    NER_LABEL_NAMES,
)

logger = logging.getLogger(__name__)

def _freeze_encoder(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder

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
    dataset_config = normalize_dataset_config(name, dataset_config)
    ds, text_fn, keep_labels = resolve_dataset(
        name=name,
        split=split,
        dataset_config=dataset_config,
        raw_dataset_root=raw_dataset_root,
        cnn_field=cnn_field,
        logger=logger,
    )
    
    if shuffle:
        ds = ds.shuffle(seed=42)

    if subset is not None:
        if subset <= 1.0:
            subset = int(len(ds) * subset)
        ds = ds.select(range(subset))

    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    encoder = _freeze_encoder(AutoModel.from_pretrained(tokenizer_name))
    ds = encode_examples(ds, tok, encoder, text_fn, max_length, keep_labels)
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

    eval_split = eval_cfg.split if hasattr(eval_cfg, "split") else "validation"
    eval_cnn_field = eval_cfg.cnn_field if hasattr(eval_cfg, "cnn_field") else None
    eval_ds, _ = get_dataset(
        split=eval_split,
        name=eval_cfg.dataset,
        subset=eval_cfg.subset,
        rebuild=cfg.data.rebuild_ds,
        shuffle=bool(eval_cfg.shuffle),
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
        shuffle=bool(eval_cfg.shuffle),
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


def get_dataset(tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
                name="cnn", split="train", dataset_config=None,
                cnn_field=None, subset=None, rebuild=False, shuffle=False):
    dataset_config = normalize_dataset_config(name, dataset_config)

    filename = dataset_cache_filename(
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
            fallback_filename = dataset_cache_filename(
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
