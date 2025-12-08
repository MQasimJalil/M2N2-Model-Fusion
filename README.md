# Hybrid CV Architecture: EfficientNet-Lite0 + TRM + M2N2

This project implements and compares a baseline EfficientNet-Lite0 model against a hybrid architecture that fuses EfficientNet-Lite0 with a Tiny Recursive Model (TRM), optimized using a simplified M2N2 (Model Merging of Natural Niches) approach. The study is conducted on the CIFAR-10 dataset to demonstrate accuracy improvements.

## Project Structure

```
.
├── configs/
│   ├── baseline.yaml         # Configuration for EfficientNet-Lite0 baseline training
│   └── fused.yaml            # Configuration for Hybrid EfficientNet-TRM training
├── scripts/
│   ├── run_train.bat         # Windows batch script to run training for all models/seeds
│   └── run_eval.bat          # Windows batch script to evaluate a single model checkpoint
├── src/
│   ├── data/
│   │   └── cifar10.py        # CIFAR-10 dataset loading and preprocessing
│   ├── models/
│   │   ├── efficientnet_lite.py # EfficientNet-Lite0 implementation (adapted for CIFAR-10)
│   │   ├── trm.py            # Tiny Recursive Model (Nano-TRM) implementation
│   │   ├── hybrid_model.py   # Combines EfficientNet-Lite0 and TRM
│   │   └── m2n2_fusion.py    # Simplified M2N2 (weight merging logic)
│   ├── utils/
│   │   ├── metrics.py        # Accuracy calculation utilities
│   │   └── seed.py           # Reproducibility seed setting
│   ├── eval.py               # Script for evaluating a trained model
│   └── train.py              # Script for training models
├── tests/
│   ├── test_models.py        # Unit tests for model shapes and forward passes
│   └── test_m2n2_fusion.py   # Unit tests for M2N2 fusion logic (slerp, merge_state_dicts)
├── checkpoints/              # Directory to save trained model checkpoints
├── cifar-10-batches-py/      # CIFAR-10 dataset (will be downloaded here)
├── requirements.txt          # Python dependencies
├── Plan.md                   # Detailed project plan
├── Results.md                # (To be generated) Experiment results, tables, and plots
└── Questions.md              # (To be generated) 30-40 questions based on the codebase
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  **Create a Python environment (recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate # On Windows
    # source venv/bin/activate # On macOS/Linux
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download CIFAR-10:**
    The dataset will be automatically downloaded to the `cifar-10-batches-py/` directory when `src/data/cifar10.py` is first run.

## Usage

### Training Models

To train the baseline EfficientNet-Lite0 model and the Hybrid EfficientNet-TRM model across multiple seeds (as defined in `scripts/run_train.bat`):

```bash
.\scripts\run_train.bat
```
Trained models will be saved in the `checkpoints/baseline/` and `checkpoints/hybrid/` directories respectively.

### Evaluating Models

To evaluate a specific trained model checkpoint:

```bash
.\scripts\run_eval.bat --model_name EfficientNetBaseline --checkpoint checkpoints\baseline\EfficientNetBaseline_best_seed_42.pth --batch_size 64
```
Replace `EfficientNetBaseline` and the checkpoint path with the desired model and file.

### Running Tests

To run the unit tests:

```bash
python -m unittest discover tests
```

## M2N2 Fusion (Post-training)

After training multiple hybrid models (e.g., `HybridEfficientNetTRM` with different seeds), you can use the `src/m2n2_fusion.py` utility to find an optimal merged model. This process involves:

1.  Loading two trained `HybridEfficientNetTRM` models.
2.  Defining an evaluation function (e.g., using `src/eval.py` on a validation set).
3.  Using `M2N2WeightMerger.find_best_merge` to find the best `mix_ratio` and `split_point` for fusing their weights.

*(Detailed instructions for running M2N2 fusion will be added in `Results.md` after initial training runs.)*

## Experiment Reproduction

The training process involves running `run_train.bat` which trains models across 3 different seeds. The `Results.md` will contain the aggregated results and instructions for reproducing the M2N2 fusion step.

## References

*   **EfficientNet.pdf**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv preprint arXiv:1905.11946.
*   **TRM.pdf**: Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv preprint arXiv:2510.04871.
*   **M2N2.pdf**: Abrantes, J. P., Lange, R. T., & Tang, Y. (2025). Competition and Attraction Improve Model Fusion. Proceedings of Genetic and Evolutionary Computation Conference 2025 (GECCO '25).
