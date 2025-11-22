# Branching Trainers

This repository hosts the branching/composite training scripts that depend on the core MoE code in `../MoE`. Run commands from here so the relative imports can locate both the MoE and RatCon projects:

```bash
cd ../branching
python src/train_composite_branching.py --config-path ../MoE/src/conf --config-name composite
```

The scripts automatically add `../MoE` and `../RatCon` to `sys.path` so they can reuse the ExpertModel and selector components.
