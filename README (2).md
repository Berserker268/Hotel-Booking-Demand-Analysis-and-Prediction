# 🏨 Hotel Booking Demand Analysis and Prediction

A machine learning project that analyzes hotel booking patterns and predicts whether a booking will be cancelled. Built using a real-world dataset of 119,390 hotel bookings, covering the full ML pipeline — from EDA and preprocessing to model comparison, clustering, and an interactive Gradio demo.

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Berserker268/Hotel_Booking_Prediction)

🚀 **Live Demo:** [Try it here](https://huggingface.co/spaces/Berserker268/Hotel_Booking_Prediction)

---

## 📌 Project Overview

Hotel booking cancellations are a major challenge for the hospitality industry, leading to revenue loss and poor resource planning. This project builds a predictive model to identify bookings likely to be cancelled, and segments customers using unsupervised learning to uncover hidden patterns.

**Target Variable:** `is_canceled` (1 = Cancelled, 0 = Not Cancelled)

---

## 📂 Dataset

- **Source:** [Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — Kaggle
- **Size:** 119,390 rows × 32 columns
- **Hotel Types:** City Hotel & Resort Hotel
- **Key Features:** `lead_time`, `arrival_date_month`, `adr`, `market_segment`, `deposit_type`, `customer_type`, `total_of_special_requests`
- **Missing Values:** `company` (112,593), `agent` (16,340), `country` (488)

---

## 🔧 Tech Stack

| Category | Libraries |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Pipelines | `Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder` |
| Demo | `gradio` |

---

## 📊 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Cancellation rate by hotel type
- Booking trends across months and seasons
- ADR (Average Daily Rate) distribution
- Lead time vs cancellation relationship

### 2. Data Preprocessing
- Dropped high-missing columns (`company`, `agent`)
- Imputed missing values in `country` and `children`
- Encoded categorical features with `OneHotEncoder`
- Scaled numerical features with `StandardScaler`
- Used `Pipeline` and `ColumnTransformer` for clean, reproducible preprocessing

### 3. Supervised Learning Models

| Model | Description |
|---|---|
| Logistic Regression | Linear baseline model |
| Random Forest Classifier | Ensemble tree-based model |
| MLP Classifier | Neural network (multi-layer perceptron) |

Models evaluated using **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **Confusion Matrix**.

### 4. Unsupervised Learning
- **K-Means Clustering** to identify distinct customer segments
- Helps understand booking behavior patterns beyond cancellation prediction

### 5. Gradio Demo
Interactive web app where you can input booking details and get a real-time cancellation prediction.

---

## 🚀 How to Run

### Option 1 — Kaggle (Recommended)
1. Open the notebook directly on Kaggle
2. Enable internet access in notebook settings
3. Run all cells

### Option 2 — Google Colab
```bash
pip install scikit-learn pandas numpy matplotlib seaborn gradio
```
Then upload and run the notebook.

### Option 3 — Local
```bash
git clone https://github.com/Berserker268/Hotel-Booking-Demand-Analysis-and-Prediction
cd Hotel-Booking-Demand-Analysis-and-Prediction
pip install -r requirements.txt
jupyter notebook "422_project_p2_2 (1).ipynb"
```

---

## 📁 Repository Structure

```
Hotel-Booking-Demand-Analysis-and-Prediction/
│
├── 422_project_p2_2 (1).ipynb    # Main notebook
├── 11.csv                         # Dataset
├── Group11_LabProject.pdf         # Project report
└── README.md                      # Project documentation
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
gradio
```

---

## 📈 Key Findings

- **Deposit type** is a strong predictor — non-refundable deposits paradoxically correlate with higher cancellation rates
- **Lead time** matters — bookings made far in advance are more likely to be cancelled
- **Online TA (Travel Agency)** market segment has the highest cancellation rate
- **Resort Hotels** have lower cancellation rates compared to City Hotels
- **Special requests** are a negative predictor of cancellation — guests with requests are more committed to their booking

---

## 🤖 Model Results

| Model | Accuracy |
|---|---|
| Random Forest | Highest |
| MLP Classifier | Competitive |
| Logistic Regression | Baseline |

> Random Forest performed best overall due to its ability to handle mixed feature types and non-linear relationships.

---

## 🔮 Future Improvements

- [ ] Add SHAP explainability for feature importance
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Try XGBoost and LightGBM
- [ ] Add time-series analysis of booking trends
- [ ] Expand Gradio demo with richer visualizations

---

## 🙋 Author

**Berserker268**
Made as part of CSE422 Lab Project.

Feel free to fork, star ⭐, and contribute!
