# PRISM main experiment code

This repository contains the complete source used for the main PRISM node-classification experiments on the A800 server. PRISM is the multiview hypergraph model described in the paper. The package keeps the full training implementation, hypergraph construction, shared and private encoders, adaptive gate, cross-view objectives, configuration search utilities, data preparation, evaluation scripts, and validation checks.

## Contents

- `models/` contains the complete PRISM and multiview hypergraph model implementations.
- `utils/` contains dataset loading and I/O utilities.
- `run_node.py` contains node-classification training, masks, metrics, and auxiliary objectives.
- `configs/node/` contains the node dataset specifications.
- `configs/prism/` contains PRISM experiment configurations.
- `experiment_runs/prism_main_experiment/` contains the full A800 main-study code, including the suite, configuration spaces, data preparation, training refinement, protocol checks, and launch scripts.
- `experiment_runs/full_benchmark/baselines.py` provides the model-independent helper imported by the main suite.

Generated checkpoints, logs, cached hypergraphs, result tables, and dataset files are excluded from this archive.

## Environment

The A800 runs used Python 3.11, PyTorch with CUDA 12.6, NumPy, SciPy, scikit-learn, pandas, PyYAML, threadpoolctl, OGB, and PyTorch Geometric. Install the packages in `requirements.txt` together with a CUDA-compatible PyTorch Geometric installation.

## Data

The public datasets are not included. Use the preparation scripts under `experiment_runs/prism_main_experiment/` to download and prepare the data into the expected local directories.

## Main experiment entry points

The main A800 suite is in `experiment_runs/prism_main_experiment/a800_suite.py`. The frozen-configuration training workflow is implemented in `a800_adaptive.py` and `a800_refine.py`. Protocol and source checks are provided by the corresponding `check_*.py` files.

Run commands from the repository root so that the relative data and configuration paths resolve correctly.

## Reproducibility

The training code preserves the validation-only configuration selection protocol, fixed dataset splits, recorded random seeds, early stopping, and single held-out test evaluation used for the main study.
