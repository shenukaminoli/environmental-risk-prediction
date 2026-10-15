# Environmental Risk Prediction and Alert System

## 1. Project Overview

This repository contains the individual research implementation of the Environmental Risk Prediction and Alert System, a component of the Smart Wearable and IoT-Driven Construction Worker Safety System.

The research focuses on predicting future environmental risk conditions at construction-site zones using historical and current environmental measurements.

The proposed approach is the **Adaptive Multi-Sensor Temporal Risk Prediction Algorithm (AMSTRPA)**. AMSTRPA is a hybrid system-level workflow that integrates existing machine-learning algorithms, temporal feature engineering, environmental risk classification, contributing-factor analysis, and configurable safety-action mapping.

The research will evaluate whether combining established prediction techniques improves future environmental risk prediction compared with individual models and a conventional threshold-based baseline.

## 2. Research Objective

To design, develop, and evaluate an environmental risk prediction workflow that processes multi-sensor environmental data and predicts the next five-minute risk category as:

- **Safe**
- **Warning**
- **Critical**

The predicted risk category is intended to support construction-site safety decisions. It does not replace professional safety judgement or approved site procedures.

## 3. Proposed Prediction Workflow

The proposed AMSTRPA workflow consists of the following stages:

1. Environmental dataset inspection and validation.
2. Data cleaning and preparation.
3. Timestamp and construction-site zone management.
4. Temporal feature engineering using historical measurements.
5. Construction of the future risk classification target.
6. Training and evaluation of individual machine-learning models.
7. Combining model predictions through a hybrid fusion method.
8. Predicting the future environmental risk category.
9. Identifying contributing environmental factors.
10. Mapping risk categories to configurable safety recommendations.
11. Evaluating the hybrid workflow against baseline approaches.

## 4. Environmental Data

The research dataset contains environmental observations from two construction-site zones.

The available environmental variables include:

- Temperature (`temperature_C`)
- Humidity (`humidity_pct`)
- Carbon dioxide (`co2_ppm`)
- Particulate matter PM2.5 (`pm25_ugm3`)
- Particulate matter PM10 (`pm10_ugm3`)
- Noise (`noise_dBA`)
- Light intensity (`light_lux`)
- Ultraviolet index (`uv_index`)

The dataset also includes timestamps, zone-related information, and an existing environmental risk label.

The exact availability and validity of each column will be verified during dataset inspection. Variables will not be invented where measurements are missing.

## 5. Prediction Target

The main prediction task is to estimate the risk category at the next five-minute observation using only information available at the current prediction time.

The input features will be derived from current and historical environmental observations. The target will be the corresponding future `overall_zone_risk` label, aligned within the same zone.

The existing labels are threshold-derived. Therefore, the experiment evaluates the prediction of configured risk categories and does not independently establish real-world hazard ground truth.

## 6. Candidate Machine-Learning Algorithms

The following existing algorithms will be investigated:

- **Random Forest:** A tree-ensemble baseline for environmental and temporal features.
- **XGBoost:** A gradient-boosted tree model for tabular environmental data.
- **LSTM:** A recurrent neural-network model for historical environmental sequences.

The proposed hybrid workflow will combine model probability outputs using a fusion strategy selected through validation experiments.

The final choice of model configuration and fusion method will be based on measured results rather than assumed performance.

## 7. Technology Stack

- Programming language: Python
- Development environment: Visual Studio Code
- Version control: Git
- Repository hosting: GitHub
- Data processing: Pandas and NumPy
- Machine learning: Scikit-learn and XGBoost
- Sequence modelling: An LSTM-compatible deep-learning framework
- Visualisation: Matplotlib and Seaborn
- Model persistence: Joblib where applicable

## 8. Repository Structure

```text
environmental-risk-prediction/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── 01_inspect_data.py
│   ├── 02_prepare_data.py
│   ├── 03_create_features.py
│   ├── 04_train_baseline.py
│   ├── 05_train_hybrid.py
│   └── 06_evaluate.py
├── results/
├── .gitignore
├── README.md
└── requirements.txt
```

Raw data, processed data, generated models, and experimental outputs are kept separate from source code. The `.gitignore` file excludes local and generated files by default.

## 9. Installation and Setup

### Prerequisites

- Python 3
- Git
- Visual Studio Code

### Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/environmental-risk-prediction.git
cd environmental-risk-prediction
```

Replace `YOUR-USERNAME` with the relevant GitHub username.

### Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The required dependencies will be updated as the implementation progresses.

## 10. Dataset Setup

Place the authorised research Excel dataset in:

```text
data/raw/
```

The raw dataset is excluded from version control by default. The dataset must be available locally before executing the inspection and preparation scripts.

## 11. Implementation Roadmap

The implementation will proceed incrementally:

1. Repository creation and initial documentation.
2. Dataset inspection and quality analysis.
3. Data preparation and timestamp validation.
4. Future-target construction for five-minute-ahead prediction.
5. Temporal feature engineering.
6. Random Forest baseline implementation.
7. XGBoost and LSTM model implementation.
8. Hybrid probability-fusion development.
9. Evaluation against individual models and a threshold-based baseline.
10. Contributing-factor analysis and configurable action mapping.
11. Experimental documentation and integration planning.

Each stage will be tested and committed separately.

## 12. Evaluation Strategy

The experimental evaluation will consider:

- Accuracy
- Precision
- Recall and sensitivity
- F1-score
- Confusion matrix
- Class-wise performance
- Alert behaviour and processing latency, where implemented

Training, validation, and testing data will be separated chronologically to reduce temporal leakage. Model selection and fusion configuration will use validation data, while the final test period will remain separate until final evaluation.

Results will be documented using the actual experimental measurements. No performance values are assumed in advance.

## 13. Research Limitations

The initial dataset contains threshold-derived risk labels. Predictive performance against these labels does not independently prove that the predicted classes correspond to verified real-world construction hazards.

The available data may not contain every environmental variable, sensor-quality indicator, or site context required by the complete proposed system. Such limitations will be documented.

The system is intended as a safety decision-support component and will not replace approved construction-site safety procedures.

## 14. Project Status

**Initial development stage**

- Research repository and documentation: planned for initial setup.
- Dataset inspection: first implementation milestone.
- Data preparation and future-target construction: pending implementation.
- Baseline and hybrid models: pending implementation.
- Experimental evaluation: pending implementation.

The project status will be updated as each component is implemented and tested.

## 15. Research Context

Research project: Environmental Risk Prediction and Alert System  
Proposed hybrid workflow: Adaptive Multi-Sensor Temporal Risk Prediction Algorithm (AMSTRPA)  
Prediction horizon: Five minutes  
Risk categories: Safe, Warning, Critical

This repository is dedicated to the environmental risk prediction component and its experimental evaluation.
