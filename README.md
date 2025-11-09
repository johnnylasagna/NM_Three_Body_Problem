# Three Body Problem — Repository Overview & Setup

This repository contains experiments and models for learning / simulating the three-body problem using three approaches: Fully Convolutional Network (FCN), Fourier Neural Operator (FNO), and Physics-Informed Neural Network (PINN).

The three-body dynamics follow Newtonian gravity, e.g.
$$
\ddot{\mathbf{r}}_i \;=\; G \sum_{j\ne i} m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{\|\mathbf{r}_j - \mathbf{r}_i\|^3}
$$

Files and notable artifacts
- Data
  - [data/TBP_dataset.csv](data/TBP_dataset.csv) — raw dataset for training/evaluation
  - [Link to download](https://www.kaggle.com/datasets/sunyuanxi/tbp-dataset?resource=download)
- FCN
  - [FCN/model_train_test_new.ipynb](FCN/model_train_test_new.ipynb) — FCN training & testing notebook
  - [FCN/testing.ipynb](FCN/testing.ipynb) — FCN testing/visualization
  - [FCN/mera_model_new_new.pth](FCN/mera_model_new_new.pth) — trained FCN model weights
- FNO
  - [FNO/main.py](FNO/main.py) — FNO training / entry script (run with Python)
  - [FNO/visualize.py](FNO/visualize.py) — visualization utilities for FNO outputs
  - [FNO/saved_model.pth](FNO/saved_model.pth) — saved FNO model
  - [FNO/saved_model_2.pth](FNO/saved_model_2.pth) — alternate saved FNO model
- PINN
  - [PINN/PINN_2d.ipynb](PINN/PINN_2d.ipynb) — PINN training notebook
  - [PINN/predictin_pinn.ipynb](PINN/predictin_pinn.ipynb) — PINN prediction notebook
  - [PINN/preprocess.ipynb](PINN/preprocess.ipynb) — data preprocessing for PINN
  - [PINN/modified_TBP_dataset.csv](PINN/modified_TBP_dataset.csv) — PINN-specific dataset
  - [PINN/pinn_2d_model.pt](PINN/pinn_2d_model.pt) — trained PINN model

Quick setup (recommended)
1. Install system requirements
   - Python 3.8+ (3.9/3.10 recommended)
   - git, (optionally) CUDA drivers if using GPU

2. Create and activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1