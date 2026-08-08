# Disease Prediction Toolkit

A modular machine-learning project for exploring healthcare classification workflows using the UCI-style heart disease dataset.

## Overview

This project demonstrates an end-to-end classical ML workflow:

**data loading → preprocessing → model training → evaluation → visualization**

It is intended for learning and experimentation with machine-learning pipelines and is **not a medical diagnostic tool**.

## Features

- Data loading and exploratory analysis
- Missing-value handling
- Categorical feature encoding
- Feature standardization
- Train/test splitting with stratification
- Multiple classification models
- Model comparison using common evaluation metrics
- Random Forest hyperparameter optimization with GridSearchCV
- Evaluation visualizations including confusion matrices and ROC curves
- Automated tests for preprocessing, model training and evaluation

## Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Optimized Random Forest

## Evaluation Metrics

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Project Structure

```text
Disease-Prediction-Toolkit/
├── Project.ipynb          # End-to-end experimentation notebook
├── heart.csv              # Dataset
├── src/
│   ├── preprocessing.py   # Data preparation pipeline
│   ├── models.py          # Model training and optimization
│   ├── evaluation.py      # Metrics and evaluation plots
│   └── visualization.py   # Exploratory visualizations
├── Requirements           # Python dependencies
├── Test File              # Current test suite
└── README.md
```

## Installation

```bash
git clone https://github.com/deadlyrps2802/disease-prediction-toolkit.git
cd disease-prediction-toolkit
pip install -r Requirements
```

## Usage

Open `Project.ipynb` in Jupyter Notebook or JupyterLab and run the workflow step by step.

The reusable functionality is organized under `src/` so the preprocessing, training and evaluation components can also be imported independently.

## Testing

The repository includes tests covering preprocessing, model training and evaluation. Run them with:

```bash
pytest
```

## Future Improvements

- Add a reproducible training script and CLI
- Add stronger pipeline abstractions with `sklearn.Pipeline`
- Add cross-validation and reproducible experiment tracking
- Add a lightweight web API for inference
- Add CI with automated tests

## Author

**Rudra Pratap Singh**

GitHub: https://github.com/deadlyrps2802
