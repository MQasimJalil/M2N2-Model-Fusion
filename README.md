# Hybrid CV Architecture: EfficientNet-Lite0 + TRM + M2N2

This project implements and compares a baseline EfficientNet-Lite0 model against a hybrid architecture that fuses EfficientNet-Lite0 with a Tiny Recursive Model (TRM), optimized using a simplified M2N2 (Model Merging of Natural Niches) approach. The study is conducted on the CIFAR-10 dataset to demonstrate accuracy improvements and enhanced robustness to various corruptions.

## Project Structure

```
.
├── configs/
│   ├── baseline.yaml         # Configuration for EfficientNet-Lite0 baseline training
│   └── fused.yaml            # Configuration for Hybrid EfficientNet-TRM training
├── scripts/
│   ├── run_train.bat         # Windows batch script to run training for all models/seeds
│   ├── perform_fusion.py     # Script to perform M2N2 weight merging on trained hybrid models
│   ├── run_eval.bat          # Windows batch script to orchestrate comprehensive evaluation
│   └── corrupt.py            # Main evaluation script for clean and corrupted data analysis (generates plots and reports)
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
│   ├── eval.py               # Script for evaluating a trained model on a given DataLoader
│   └── train.py              # Script for training models
├── tests/
│   ├── test_models.py        # Unit tests for model shapes and forward passes
│   └── test_m2n2_fusion.py   # Unit tests for M2N2 fusion logic (slerp, merge_state_dicts)
├── checkpoints/              # Directory to save trained model checkpoints
├── cifar-10-batches-py/      # CIFAR-10 dataset (will be downloaded here)
├── requirements.txt          # Python dependencies for pip
├── environment.yml           # Conda environment definition for reproducible setup
├── Plan.md                   # Detailed project plan
├── Results.md                # (To be generated) Experiment results, tables, and plots
└── Questions.md              # (To be generated) 30-40 questions based on the codebase
```

## Setup (Reproducible Environment)

It is highly recommended to use a dedicated Python environment to manage dependencies.

### Using Conda (Recommended for ease of installation)

If you have Anaconda or Miniconda installed:

1.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the environment:**
    ```bash
    conda activate cv-proj
    ```

### Using Pip

1.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Download CIFAR-10 Dataset

The dataset will be automatically downloaded to the `cifar-10-batches-py/` directory when `src/data/cifar10.py` is first run during training or evaluation.

## Usage

### 1. Training Hybrid Models (with Fixed Initialization for M2N2)

To enable effective M2N2 fusion, models must share a common initialization point. The `train.py` script has been modified to use a fixed seed for model weight initialization while varying the seed for data shuffling and training dynamics.

To train the Hybrid EfficientNet-TRM model across multiple seeds (as defined in `scripts/run_train.bat`):

```bash
.\scripts\run_train.bat
```
Trained models will be saved in the `checkpoints/hybrid/` directory.

### 2. Performing M2N2 Fusion

After training at least two (ideally three or more) hybrid models, you can use the `scripts/perform_fusion.py` utility to iteratively merge their weights. This finds an optimal combined model that often outperforms individual constituents.

```bash
python scripts/perform_fusion.py
```
The fused model checkpoint will be saved as `checkpoints/hybrid/HybridEfficientNetTRM_fused.pth`.

### 3. Comprehensive Evaluation & Robustness Analysis

The `scripts/run_eval.bat` script orchestrates a full evaluation suite, including performance on clean CIFAR-10 data and robustness against various corruptions (Gaussian Noise, Defocus Blur, Frost) at different severity levels. It also generates visual aids and a detailed report.

```bash
.\scripts\run_eval.bat
```

**Outputs:**
*   `results/evaluation_results.txt`: A detailed text report of all accuracy metrics.
*   `results/dashboard.png`: A single PNG image showing comparative performance on clean and corrupted data.
*   `results/viz_*.png`: Visualizations of model predictions on sample corrupted images, showing true vs. predicted labels.

### Running Unit Tests

To run the project's unit tests:

```bash
python -m unittest discover tests
```

## Key Findings (Summary of Results)

*(This section will be populated with a summary of the project's findings after initial runs.)*

Preliminary findings indicate:

*   **Clean Data Performance:** The M2N2-fused Hybrid EfficientNet-TRM model maintains competitive accuracy on clean CIFAR-10 data (e.g., 0.8841 vs Baseline Average 0.8885).
*   **Robustness to Noise and Frost:** The Hybrid model demonstrates significantly improved robustness to high-frequency corruptions like Gaussian Noise and Frost. It consistently outperforms the Baseline EfficientNet, with improvements reaching over **+3%** at higher severities. This suggests the TRM's recursive mechanism aids in filtering or re-interpreting noisy features.
*   **Defocus Blur Trade-off:** The Hybrid model shows reduced performance or a slight gain compared to the Baseline on Defocus Blur, especially at lower severities. This indicates a potential trade-off in handling severe loss of high-frequency information, where the recursive processing might struggle to recover details that are fundamentally absent.

These results highlight the specific conditions under which a Hybrid-TRM architecture, combined with M2N2 fusion, can enhance model robustness.

## References

*   **EfficientNet.pdf**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv preprint arXiv:1905.11946.
*   **TRM.pdf**: Jolicoeur-Martineau, A. (2025). Less is More: Recursive Reasoning with Tiny Networks. arXiv preprint arXiv:2510.04871.
*   **M2N2.pdf**: Abrantes, J. P., Lange, R. T., & Tang, Y. (2025). Competition and Attraction Improve Model Fusion. Proceedings of Genetic and Evolutionary Computation Conference 2025 (GECCO '25).
