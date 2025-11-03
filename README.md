#  A Synergistic Strategy for Data-constrained Deep Learning in Materials Science

This is a attention pooling with multi-task learning experiment repository based on the Crystal Graph Convolutional Neural Network (CGCNN), containing training scripts, grid search scripts, logging tools, and small utilities for plotting and data merging. This project is used to simultaneously predict the work function and band gap of materials, and supports conditional loss (based on metal/semiconductor classification) and several optional auxiliary tasks.

## Directory Structure (Simplified)

- `main.py` — Main training/validation/testing script (CLI-driven).
- `grid_search.py` — Grid search/batch experiment script for running `main.py` with different hyperparameter combinations.
- `logger_utils.py` — Logging helper functions (create log files, format training/validation metrics).
- `root_dir/` — Example data directory (contain *.cif files, id_prop.csv and all.csv(whose second line represents the category of the material)).

----

## Key Features

- Supports multi-task regression (work function + band gap) as well as classification tasks.
- Supports conditional band gap loss (based on metal/semiconductor classification) and multiple optional auxiliary tasks.
- Supports fixed or adaptive task weights.
- Writes metrics for each experiment to the `result/` directory and generates log files with detailed information (`result/log/`).
- Records the time taken for each epoch during training and generates per-split and overall timing CSV reports.

----

## Environment and Dependencies (Recommended)

Please run in a Python 3.8+ environment. Typical dependencies include (please adjust versions according to your environment):

- torch (PyTorch)
- numpy
- pandas
- scikit-learn
- matplotlib
- scipy

Can be installed with pip (example):

```powershell
pip install torch numpy pandas scikit-learn matplotlib scipy
```

Note: This repository depends on the `cgcnn` module (containing `cgcnn.data` and `cgcnn.model`). Please ensure this package is in the Python path (it may be a submodule within the repository or a separately installed package).

----

## Quick Start: Usage Examples

In PowerShell (with the working directory as the project root):

1) Basic training (uses SGD by default) example:

```powershell
python .\main.py .\root_dir
```


2) Example run with common parameters:

```powershell
python .\main.py .\root_dir --scheduler step --pooling attention --epochs 120 --lr 0.011 --weight-decay 0.0 --task2-weight 0.5 --auxiliary-tasks --aux-task3-weight 0.05
```

3) Run all combinations in grid search (example):

```powershell
python .\grid_search.py .\root_dir --num-splits 1 --epochs 120
```

`grid_search.py` will sequentially call `main.py` and generate `result/all_combinations_summary_<timestamp>.csv` to summarize the metrics for each combination.

----

## Important Command-Line Arguments for `main.py`

- `data_options` — Required argument, usually the root directory for the data (e.g., `root_dir`).
- `--auxiliary-tasks` — Enable auxiliary tasks (electronegativity variance, formation energy, first ionization energy average).
- `--aux-task1-weight/--aux-task2-weight/--aux-task3-weight` — Set the loss weights for the three auxiliary tasks respectively (default 0.1).
  - Note: `--aux-task3-weight` controls the contribution of auxiliary task 3 (first ionization energy average) to the total loss. Setting it to `0` is equivalent to disabling the effect of this task's loss on the training objective, but the task's data/output can still be retained (just not included in the loss).
- `--conditional-loss` — Enable conditional band gap loss based on metal/semiconductor classification (requires parameters like `--classification-weight`, `--metal-weight`, `--semiconductor-weight`).
- `--adaptive-weights` — Enable the adaptive task weighting mechanism (automatically adjusts task1/task2 weights during training).
- `--optim` — Optimizer selection, default is `SGD`. The code currently defaults to using `optim.SGD(...)`. If you want to use Adam, you can complete the branch in `main.py` or pass `--optim Adam` and ensure the script creates an Adam optimizer in that branch.

For more parameters, see the `argparse` definitions at the top of the `main.py` file.

----

## Output, Logs, and Timing

- All experiment outputs are written to the `result/` folder. This includes:
  - Logs: `result/log/{run_id}_split{n}_<timestamp>.log` (created by `logger_utils.setup_logger`).
  - Metrics CSV: `result/{run_id}_metrics.csv` (contains per-epoch or per-split metrics, depending on how the script writes them).
  - MAE Summary: `result/{run_id}_mae.csv` (contains MAE for each split, as well as the average/standard deviation).
  - Trained Model: `result/{run_id}_trained_model.pt` (contains the `state_dict`, normalizer states, and `args`).
  - Prediction/Visualization Output: `result/predictions/` (`main.py` creates this directory and can write prediction results to it after training).

- Timing: `main.py` has integrated timing for each epoch:
  - Each split will write to `{run_id}_split{n}_timing.csv` (including time per epoch, average time per epoch, total training time, etc.).
  - An overall summary across splits will be written to `{run_id}_overall_timing_summary.csv` (when running multiple splits).

These timing reports help in estimating training time and managing large-scale experiments.

----

## Logging Tool `logger_utils.py`

This module is responsible for generating timestamped log files based on `run_id` and `split`, and provides formatted output functions:

- `setup_logger(run_id, split, log_dir)` — Creates and returns a logger (files are saved to `result/log/`).
- `log_train_metrics(logger, epoch, metrics)` — Writes training phase metrics to the log.
- `log_val_metrics(logger, epoch, metrics, phase)` — Writes validation/test phase metrics to the log.

Other scripts like `plot_gap_predictions.py` and `merge_work_band.py` depend on or work with these log/result outputs, but they are independent tools themselves.

----


