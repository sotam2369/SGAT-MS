# SGAT: A SAT-based Graph Attention Network

Spotlight Poster – NeurIPS 2025  
Poster: https://neurips.cc/virtual/2025/poster/116771

SGAT couples t-norm logic with graph attention to tackle weighted and unweighted MaxSAT through differentiable optimization. The model mirrors greedy distributed local search while remaining end-to-end trainable and GPU friendly.

## Highlights
- SAT-aware attention layers that inject clause satisfaction margins directly into message passing.
- Unified training across MaxSAT Evaluation benchmarks from 2018 to 2024 with shared weighted/unweighted handling.
- Baselines for classical GATs and solver wrappers, enabling direct comparisons and ablations.
- Pretrained checkpoints and plotting utilities for fast inspection of loss and evaluation curves.

## Poster and Paper
- Title: **Graph-Based Attention for Differentiable MaxSAT Solving**
- Spotlight presentation at NeurIPS 2025. Camera-ready preprint and supplementary material will be linked here when released.
- Slides and additional poster assets will be added after the conference.

## Environment Setup
- Python 3.10 or later is recommended.
- GPU acceleration is optional but provides substantial speedups; install CUDA-enabled builds of PyTorch if available.
- Core Python packages:
  - `torch`, `torch_geometric`, `python-sat`, `numpy`, `tqdm`, `matplotlib`

Example setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torch_geometric python-sat numpy tqdm matplotlib
```

## Data Preparation
- Raw competition benchmarks can be downloaded and filtered with `script/tools/download_maxsat_data.sh`.
- The script fetches MaxSAT Evaluation archives, normalizes file layouts, and produces selector-curated subsets inside `MaxSAT_Dataset/`.
- For custom datasets, adapt `tools/selector.py` to extract clause structures and weights compatible with SGAT preprocessing.

```bash
bash script/tools/download_maxsat_data.sh
```

## Quick Start
1. Initialize folders, symbolic links, and sanity checks:
   ```bash
   bash startup.sh
   ```
2. Launch the full end-to-end pipeline (startup, data prep, training, evaluation):
   ```bash
   bash all_in_one.sh
   ```
3. Train SGAT with custom arguments:
   ```bash
   python src/main.py \
     --dataset-train dataset/ms2018_train_2.0-0_nfs_nsl.pt \
     --dataset-test dataset/ms2018_test_2.0-0_nfs_nsl.pt \
     --epochs 1000 \
     --layers 6 \
     --heads 2 \
     --t-norm godel \
     --cuda cuda:0 \
     --dir plots/SGAT/Godel
   ```

Key flags in `src/main.py`:
- `--t-norm`: choose among `godel`, `product`, `lukasiewicz`, `soft_godel`, or `einstein`.
- `--use-gat`: swap SGAT layers for standard GATConv baselines.
- `--best-weights`, `--output-epochs`, `--finish-round`: control checkpointing and early stop heuristics.

## Reproducing Spotlight Experiments
- `script/experiments/experiment1.sh`: reproduces the SGAT vs. classical GAT comparison on MaxSAT 2018 subsets, including automated plotting.
- `script/experiments/experiment2.sh`: extends ablations over t-norm choices, normalization, and dropout schedules.
- Each script leverages the shared `run_allow_fail` helper so longer sweeps continue despite occasional timeouts.
- Logs, checkpoints, and figures are stored in `plots/`, organized by model family and training ID.

## Evaluation and Analysis
- `tools/plot_results_combiner.py` merges results across runs, producing publication-grade figures with shaded confidence intervals.
- `tools/selector.py` filters solver outputs and dataset artifacts, enabling fair benchmarking across weighted and unweighted tracks.
- Pretrained weights in `pretrained_models/` can be loaded through `src/manager/folder_manager.py` for zero-shot evaluation.

## Repository Overview
```
├── dataset/                  # Serialized PyTorch datasets and splits
├── pretrained_models/        # Spotlight-ready checkpoints
├── script/
│   ├── experiments/          # Reproduction scripts and ablation sweeps
│   └── tools/                # Dataset download and preparation utilities
├── solvers/                  # Third-party MaxSAT solvers (archived)
├── src/
│   ├── manager/              # Folder and dataset orchestration
│   ├── solver/               # Interfaces to external MaxSAT solvers
│   ├── utils/                # Data processing helpers
│   └── sgat.py               # Core SGAT model components
├── tools/                    # Plotting and analysis helpers
├── vba/                      # Benchmark cost references (VBA scores 2020–2024)
└── plots/                    # Training curves and combined figures
```

## Citation
If you use SGAT in your research, please cite the NeurIPS 2025 spotlight paper. A BibTeX entry will be provided alongside the camera-ready manuscript.

## Contact
Questions and feedback are welcome via the NeurIPS 2025 virtual poster page or the forthcoming project site.
