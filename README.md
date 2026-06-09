# 5G User Prediction

This repository contains a machine learning project for predicting whether a telecommunication user is a 5G user based on their basic information and telecom usage data (such as billing, data usage, active behaviors, and regional info).

## Project Overview

The task is framed as a **binary classification** problem. The dataset contains 60 features, including:
- `cat_0` to `cat_19`: Categorical (discrete) features.
- `num_0` to `num_37`: Continuous (numerical) features.
- `target`: The target variable (whether the user is a 5G user).

The primary evaluation metric for this project is **AUC (Area Under ROC Curve)**.

## Key Features & Techniques

- **Data Preprocessing & EDA**:
  - Handled fractional anomalies in categorical features (e.g., `cat_12`).
  - Applied **Double Log Smoothing** (`log1p(log1p(x))`) to heavily right-skewed (long-tailed) numerical features to normalize their distributions.
  - Missing values imputation (median for numerical features, `-1` for categorical features).

- **Modeling Strategy**:
  - **LightGBM**: A powerful gradient boosting framework optimized for tabular data. Handled imbalanced target data using balanced class weights and tuned parameters (e.g., `num_leaves=64`, `learning_rate=0.05`, `reg_lambda=10`).
  - **Tabular Neural Network**: A deep learning approach tailored for tabular data. Utilized **Entity Embeddings** to map high-dimensional sparse categorical features into dense low-dimensional vectors, combined with a Multi-Layer Perceptron (MLP) for continuous features.
  
- **Model Ensemble**:
  - Used a **Weighted Average** strategy to combine the predictions of LightGBM and the Tabular Neural Network (Weights: LightGBM 0.39, NN 0.61), which achieved higher AUC scores than single models.

## Repository Structure

```text
├── src/
│   ├── lgb_nn_ensemble.ipynb   # Main notebook (data processing, modeling, ensemble)
│   ├── 数据探索.ipynb            # Exploratory Data Analysis (EDA)
│   └── train.zip               # Training dataset (compressed; contains train.csv)
├── 人工智能小组作业.pdf            # Project task description
└── README.md                   # This file
```

> **Note on the dataset**: The original `train.csv` is too large to be tracked by Git, so it has been compressed into `src/train.zip` and the raw `train.csv` is excluded via `.gitignore`.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `lightgbm`
- `torch` (PyTorch)

You can install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm torch
```

### Data Preparation

The dataset is shipped as a compressed archive `src/train.zip` (the raw `train.csv` is ignored by Git because of its size). Before running the notebooks, extract the CSV from the zip:

**Windows (PowerShell):**
```powershell
Expand-Archive -Path src\train.zip -DestinationPath src\
```

**Windows (File Explorer):**
1. Navigate to the `src/` folder.
2. Right-click `train.zip` → **Extract All…** → choose `src/` as the destination → **Extract**.

**Linux / macOS:**
```bash
unzip src/train.zip -d src/
```

After extraction, you should see `src/train.csv`, and the notebooks can be run directly.

### Running the Code

1. Clone this repository to your local machine.
2. Extract the dataset as described in **Data Preparation** above.
3. Open and run the Jupyter notebooks in the `src/` directory.