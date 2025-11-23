Hydra: Branching Rationales
===========================

Branching composite rationale trainer built on top of SBERT encoders. A selector learns token-level rationales, an expert mixture routes selected tokens to latent factors, and a tree of selector/expert pairs is expanded stage-by-stage to specialize on harder subsets of the data. Checkpoints and metrics are produced with Hydra + dora for reproducible experiments.

Project layout
--------------
- `src/train.py`: Hydra entrypoint that drives staged training or evaluation-only runs.
- `src/tree.py`: Branching tree that duplicates selector/expert leaves and routes examples through the hierarchy.
- `src/data.py`: Dataset utilities, HF dataset loading, and cached-embedding loader.
- `src/conf/composite.yaml`: Default experiment configuration (datasets, optimizers, routing, and Slurm notes).

Setup
-----
1. Python 3.10+ virtualenv recommended:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
2. Install dependencies (pick the right PyTorch wheel for your CUDA/CPU setup):
   ```bash
   pip install torch torchvision torchaudio  # or the CUDA-specific wheel
   pip install transformers datasets sentence-transformers hydra-core dora-search prettytable tqdm attrs
   ```

Prepare cached datasets
-----------------------
The trainer expects precomputed SBERT embeddings saved with `datasets.save_to_disk` under `data/`. For the default WikiANN English setup, build the train/validation/test caches once:
```bash
python - <<'PY'
from pathlib import Path
from src.data import build_dataset, _dataset_cache_filename

def build(split):
    ds, _ = build_dataset(
        name="wikiann",
        split=split,
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=256,
        dataset_config="en",
        subset=1.0,
    )
    path = Path("data") / _dataset_cache_filename("wikiann", split, 1.0, dataset_config="en")
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(path)
    print(f"Saved {split} to {path}")

for split in ["train", "validation", "test"]:
    build(split)
PY
```
Other supported datasets: `cnn`, `conll2003`, `wnut`, `ontonotes`, `bc2gm`, and `framenet` (some require pre-downloaded raw files; see in-code warnings). If you are running offline, set the HF cache env vars shown under `slurm.setup` in `src/conf/composite.yaml`.

Running training
----------------
- Default staged training on WikiANN:
  ```bash
  python -m src.train
  ```
  This grows the tree over 5 stages (`train.stage_epochs`) and saves `branching_composite.pth` plus logs to `train_composite_branching.log` inside the Hydra run directory.
- Evaluation-only on an existing checkpoint:
  ```bash
  python -m src.train eval.eval_only=true
  ```
  Make sure `branching_composite.pth` is present in the working directory (Hydra changes `cwd` to the run dir).

Config tweaks
-------------
- Override any Hydra config at the CLI, e.g.:
  ```bash
  python -m src.train device=cpu expert_model.expert.num_experts=3 data.train.dataset=conll2003 data.eval.dataset=conll2003
  ```
- Adjust stage schedule with `train.stage_epochs`, selector threshold with `train.selector_threshold`, or turn off shuffle via `data.train.shuffle=false`.
- Slurm parameters under `slurm.*` are provided as guidance for cluster launches but are not invoked automatically by the script.

Artifacts
---------
- `branching_composite.pth`: saved selector/expert state dicts keyed by leaf path.
- `train_composite_branching.log`: human-readable log mirrored to stdout.
- Hydra/dora run folders under `outputs/` capture config, logs, and signatures. Run directory is printed at startup.
