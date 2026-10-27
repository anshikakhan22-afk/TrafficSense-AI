# TrafficSense AI: Traffic Congestion Prediction Using Artificial Neural Networks (ANN)

TrafficSense AI is a complete, beginner-friendly machine learning project designed to predict urban highway traffic congestion levels using an **Artificial Neural Network (ANN)**.

Built as an academic data science mini-project for an **MBA in Data Science & AI**, this application trains a multi-layer feed-forward neural network (`MLPClassifier`) on 6 years of historical interstate traffic and weather observations, evaluates it against a strict chronological hold-out dataset, and deploys it as an interactive **Streamlit** dashboard.

---

## Table of Contents

1. [Problem Statement & Business Context](#1-problem-statement--business-context)
2. [Project Objectives](#2-project-objectives)
3. [Technology Stack](#3-technology-stack)
4. [Dataset Details & Source](#4-dataset-details--source)
5. [Target Label Definition](#5-target-label-definition)
6. [Project Structure](#6-project-structure)
7. [Installation & Setup (Windows / VS Code)](#7-installation--setup-windows--vs-code)
8. [Training the Neural Network](#8-training-the-neural-network)
9. [Running the Streamlit Web Application](#9-running-the-streamlit-web-application)
10. [ANN Architecture Explained](#10-ann-architecture-explained)
11. [Actual Measured Model Results](#11-actual-measured-model-results)
12. [Visualizations & Outputs](#12-visualizations--outputs)
13. [Limitations & Ethical Considerations](#13-limitations--ethical-considerations)
14. [MBA Viva Defense: Key Questions & Simple Explanations](#14-mba-viva-defense-key-questions--simple-explanations)
15. [GitHub Publication Guide](#15-github-publication-guide)

---

## 1. Problem Statement & Business Context

Traffic congestion imposes enormous economic and environmental costs on modern cities:
- **Economic Loss**: Millions of commuting hours lost and delayed commercial logistics.
- **Environmental Impact**: Excess fuel consumption and elevated greenhouse gas emissions from idling vehicles.
- **Municipal Planning**: City authorities require predictive insights to plan variable toll pricing, dispatch emergency services, and schedule maintenance without causing gridlock.

**TrafficSense AI** tackles this problem by forecasting whether highway traffic will be **Low**, **Moderate**, or **High** given future time, calendar, and forecasted weather conditions.

---

## 2. Project Objectives

- **Data Exploration**: Clean, examine, and visualize multi-year real-world hourly traffic data.
- **Leakage Prevention**: Implement a strict chronological split (past data for training, future data for testing).
- **Neural Network Modeling**: Build and train a Multi-Layer Perceptron (MLP) from Scikit-Learn.
- **Rigorous Evaluation**: Measure accuracy, precision, recall, macro F1-score, and confusion matrix on unseen future records.
- **Interactive UI**: Develop a Streamlit application enabling scenario testing, confidence analysis, and visualization review.

---

## 3. Technology Stack

- **Core Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`MLPClassifier`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
- **Model Persistence**: Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Web Interface**: Streamlit
- **Version Control**: Git, GitHub

---

## 4. Dataset Details & Source

The project uses the **Metro Interstate Traffic Volume** dataset:
- **Repository**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)
- **Citation**: Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. DOI: `10.24432/C5X60B`.
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Corridor**: Westbound Interstate 94 (I-94) connecting Minneapolis and St. Paul, Minnesota, USA.
- **Record Span**: Hourly records from **October 2, 2012 to September 30, 2018** (48,187 total rows).

### Feature Dictionary

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `date_time` | Datetime | Hourly timestamp of observation |
| `holiday` | Categorical | US national holidays (e.g. Thanksgiving, Labor Day, or None) |
| `temp` | Numeric | Average ambient temperature in Kelvin |
| `rain_1h` | Numeric | Amount in mm of rain that occurred in the hour |
| `snow_1h` | Numeric | Amount in mm of snow that occurred in the hour |
| `clouds_all` | Numeric | Percentage of cloud cover (0–100%) |
| `weather_main` | Categorical | Short description of current weather (Clear, Clouds, Rain, Snow, etc.) |
| `traffic_volume` | Numeric | Hourly vehicle count (Target source, excluded from inputs) |

---

## 5. Target Label Definition

The original raw dataset records numeric hourly traffic counts (`traffic_volume`), not subjective congestion text. To model traffic levels scientifically:

1. **Threshold Derivation**: The training dataset volume distribution was divided into three equal quantiles (terciles):
   - **Low Congestion**: Hourly volume $\le 2,154$ vehicles/hour
   - **Moderate Congestion**: $2,154 <$ volume $\le 4,555$ vehicles/hour
   - **High Congestion**: Hourly volume $> 4,555$ vehicles/hour
2. **Leakage Protection**: The quantiles were calculated **strictly on the 70% training partition**. The validation and test sets used these learned thresholds without recalculation.
3. **Feature Separation**: The raw `traffic_volume` column is **never** provided to the model during inference. The ANN predicts the congestion level purely from time and weather inputs.

---

## 6. Project Structure

```text
TrafficSense-AI/
│
├── app.py                   # Interactive Streamlit Web Application
├── train_model.py           # Dataset loading, chronological split, ANN training & evaluation
├── data_preprocessing.py    # Feature engineering, imputation, and category transformation
├── test_traffic_sense.py    # Automated unit and integration test suite (7 tests)
├── requirements.txt         # Compatible Python dependencies
├── README.md                # Comprehensive project documentation & viva guide
├── .gitignore               # Ignores virtual environment, cache, and local files
│
├── data/
│   ├── Metro_Interstate_Traffic_Volume.csv.gz # UCI dataset file
│   └── README.md            # Dataset origin and instructions
│
├── models/
│   ├── traffic_sense_ann.joblib # Trained Scikit-Learn pipeline
│   ├── model_metadata.json      # Actual measured metrics from training
│   └── README.md
│
├── outputs/                 # Evaluation charts generated during training
│   ├── category_distribution.png
│   ├── traffic_by_hour.png
│   └── confusion_matrix.png
│
├── src/                     # Source modules placeholder
└── images/                  # Application screenshots for portfolio
```

---

## 7. Installation & Setup (Windows / VS Code)

### Prerequisites

Verify that Python and Git are installed:

```powershell
python --version
git --version
```

### 1. Clone or Open the Repository

Open the folder `TrafficSense-AI` in Visual Studio Code.

### 2. Create the Virtual Environment

```powershell
python -m venv .venv
```

### 3. Install Dependencies

Using `.\.venv\Scripts\python.exe` directly executes commands inside the virtual environment without needing to change PowerShell script execution policies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 8. Training the Neural Network

Execute the training script:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The training process:
1. Loads `Metro_Interstate_Traffic_Volume.csv.gz` from `data/`.
2. Splits data chronologically:
   - **Training Set (70%)**: 33,730 observations (2012–2016)
   - **Validation Set (15%)**: 7,228 observations (2016–2017)
   - **Testing Set (15%)**: 7,229 observations (2017–2018)
3. Fits the Scikit-Learn preprocessing pipeline (`StandardScaler` + `OneHotEncoder`).
4. Trains the `MLPClassifier` with early stopping.
5. Evaluates on the held-out test set and saves the model and charts.

---

## 8b. Running the Automated Test Suite

Run the built-in test suite to verify data loading, feature engineering, leak-free thresholding, model inference, and output files:

```powershell
.\.venv\Scripts\python.exe test_traffic_sense.py
```

---

## 9. Running the Streamlit Web Application

Launch the local web dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app will start and open automatically in your browser at:
`http://localhost:8501`

### Application Features:
- **🏠 Project Overview**: Executive summary, problem definition, and workflow diagrams.
- **🔮 Interactive Prediction**: Input weather and time parameters, use one-click scenario presets (Morning Rush, Midnight, Snowstorm), and view real-time predictions with class confidence bars.
- **📊 Dataset & Visualizations**: Interactive review of hourly traffic trends, target class balance, and the test confusion matrix.
- **🧠 Model Architecture & Metrics**: Hyperparameter details, precision/recall/F1 metrics table, and academic limitations.

---

## 10. ANN Architecture Explained

The model is a feed-forward **Multi-Layer Perceptron (MLP)**:

```text
[Input Features: 9]
  │  (hour, day_of_week, month, temp, rain, snow, clouds, holiday, weather)
  ▼
[Preprocessing Pipeline: Imputation + Scaling + One-Hot Encoding]
  │
  ▼
[Hidden Layer 1: 64 Neurons] ───► Activation: ReLU
  │
  ▼
[Hidden Layer 2: 32 Neurons] ───► Activation: ReLU
  │
  ▼
[Output Layer: 3 Neurons]    ───► Activation: Softmax (Low, Moderate, High)
```

### Core Concepts:
- **Neurons & Layers**: Neurons compute weighted sums of inputs ($z = \sum w_i x_i + b$) and pass them through non-linear activation functions.
- **ReLU (Rectified Linear Unit)**: $f(z) = \max(0, z)$. Prevents vanishing gradient issues and allows the network to learn complex non-linear highway traffic behaviors.
- **Optimizer (Adam)**: Adapts individual learning rates for each parameter based on first and second moments of gradients.
- **Loss Function**: Multi-class Cross-Entropy Loss measuring divergence between predicted probability distributions and true labels.
- **Early Stopping**: Halts training when validation score does not improve for 15 consecutive iterations, preventing overfitting to training data.

---

## 11. Actual Measured Model Results

The following metrics were calculated from the **unseen held-out test partition** (7,229 chronological observations):

| Congestion Category | Precision | Recall | F1-Score | Support (Sample Count) |
| :--- | :---: | :---: | :---: | :---: |
| **Low Congestion** | **96.2%** | **91.3%** | **93.7%** | 2,384 |
| **Moderate Congestion** | **86.4%** | **83.4%** | **84.9%** | 2,349 |
| **High Congestion** | **87.4%** | **94.4%** | **90.8%** | 2,496 |
| **Overall Model Accuracy** | — | — | **89.8%** | 7,229 |
| **Macro Average** | **90.0%** | **89.7%** | **89.8%** | 7,229 |
| **Validation Accuracy** | — | — | **89.1%** | 7,228 |

### Confusion Matrix (Test Set)

```text
                     Predicted: Low   Predicted: Moderate   Predicted: High
Actual: Low               2,176               170                  38
Actual: Moderate             86             1,960                 303
Actual: High                  0               139               2,357
```

**Key Takeaways**:
- **0 High vs Low Misclassifications**: The model never once mistook a peak rush-hour traffic condition for empty highway conditions ($0$ false negatives for High when Low was predicted).
- **Strong Precision & Recall**: Low Congestion precision is 96.2%, and High Congestion recall is 94.4%.

---

## 12. Visualizations & Outputs

All generated charts are automatically saved in the [`outputs/`](outputs/) directory:
1. `category_distribution.png`: Shows balanced distribution across the derived tercile classes.
2. `traffic_by_hour.png`: Visualizes morning (7–9 AM) and evening (4–6 PM) commuting peaks.
3. `confusion_matrix.png`: Heatmap demonstrating model classification density along the true diagonal.

---

## 13. Limitations & Ethical Considerations

1. **Proxy Target**: Volume counts serve as a proxy for congestion. True gridlock may occasionally exhibit lower vehicle throughput if vehicles are completely stopped.
2. **Geographic Scope**: Dataset reflects one specific westbound corridor of I-94 in Minnesota. Local geography, highway ramps, and driver behavior vary across cities.
3. **Academic Scope**: Demonstrates neural network capability on open government/research data; not a replacement for live municipal GPS probe data.

---

## 14. MBA Viva Defense: Key Questions & Simple Explanations

Here are concise answers for your college presentation and viva:

#### Q1: Why did you use an Artificial Neural Network (ANN) instead of simple linear regression?
> **Answer**: Traffic patterns are highly non-linear. For example, traffic volume does not increase linearly throughout the day; it spikes sharply during morning and evening rush hours and drops precipitously at night. Furthermore, weather effects interact non-linearly with time (e.g., snow at 8 AM has a vastly different impact than snow at 2 AM). An ANN with hidden layers and ReLU activations can model these intricate multi-variable interactions naturally.

#### Q2: Why did you split the dataset chronologically instead of using random train_test_split?
> **Answer**: Traffic is a time-series phenomenon. If we shuffle data randomly, observations from the same day or week would appear in both training and test sets, causing **lookahead bias / data leakage** where the model "memorizes" conditions. A chronological 70/15/15 split trains strictly on earlier years (2012–2016) and evaluates on later, unseen future years (2017–2018), simulating real-world predictive deployment.

#### Q3: Why is traffic volume omitted from the input features?
> **Answer**: Including traffic volume as an input feature would constitute **target leakage** because our target classes (Low, Moderate, High) were derived from volume terciles. If the model had access to volume, it would be a trivial lookup rather than a true forecast. At prediction time, a user only knows the time, day, and weather forecast, not the future volume.

#### Q4: What does the confusion matrix tell us about business risk?
> **Answer**: In logistics and municipal dispatch, the most dangerous error is predicting "Low Congestion" when traffic is actually "High" (a False Negative for congestion). Our confusion matrix reveals that this occurred **0 times** out of 2,496 actual high-congestion events, making the system highly dependable for avoiding unexpected delays.

---

## 15. GitHub Publication Guide

To push this repository to GitHub:

### Step 1: Initialize Git and Check Status

```powershell
git init
git status
```

### Step 2: Add Files and Commit

```powershell
git add .
git commit -m "Initial commit: TrafficSense AI complete project"
```

### Step 3: Create GitHub Repository & Push

1. Log into your GitHub account at [github.com](https://github.com).
2. Click **New Repository** and name it `TrafficSense-AI`.
3. Keep it Public and do **not** check "Add a README" (we already have a complete one).
4. Run the following commands in your PowerShell terminal:

```powershell
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/TrafficSense-AI.git
git branch -M main
git push -u origin main
```

---

## 16. Author Information

- **Student**: MBA Candidate (Data Science & AI)
- **Course**: Machine Learning & Predictive Analytics
- **Platform**: Google Antigravity & Visual Studio Code (Windows)
