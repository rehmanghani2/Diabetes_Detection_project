# Diabetes Prediction & Analysis Dashboard

An interactive, premium Streamlit dashboard to predict diabetes risks in patients and analyze the underlying clinical dataset. The dashboard loads a trained **Decision Tree Classifier** model and its preprocessing scaler to perform real-time risk classification.

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.10+ installed on your system.

### 1. Install Dependencies
Install all the required Python packages:
```bash
pip install streamlit scikit-learn pandas matplotlib seaborn jinja2
```

### 2. Run the Dashboard
Navigate to the root project directory and start the Streamlit server:
```bash
streamlit run app.py
```
After starting the server, open your web browser and navigate to:
**[http://localhost:8501](http://localhost:8501)**

---

## 📂 Project Structure

```text
Diabetes_prediction_project/
├── app.py                     # Main Streamlit dashboard application
├── README.md                  # Project documentation (this file)
└── All Docs/
    ├── models/
    │   ├── best_model.pkl     # Saved Decision Tree model (joblib format)
    │   ├── scaler.pkl         # Fitted StandardScaler preprocessing pipeline
    │   └── feature_names.pkl  # List of features used during training
    ├── files/
    │   ├── pima.csv           # Baseline Pima Indians Diabetes Dataset
    │   └── model_comparison_results.csv  # Performance metrics for compared models
    ├── Notebooks/
    │   ├── Diabetes_Prediction_ML_Project.ipynb          # Model exploration notebook
    │   └── Diabetes_Prediction_ML_Project_Kaggle.ipynb   # Model training & serialization notebook
    └── result/
        ├── fig1_class_distribution.png
        ├── fig2_feature_distributions.png
        ├── fig3_correlation_heatmap.png
        ├── fig4_accuracy_comparison.png
        ├── fig5_prf1_comparison.png
        ├── fig6_training_time.png
        └── fig7_confusion_matrix.png
```

---

## 🩺 Clinical Features Reference

The model utilizes **7 clinical features** to assess patient health status. *Note: `Pregnancies` was excluded from this analysis to focus strictly on metabolic, genetic, and age-related indices.*

| Feature Name | Description | Clinical Baseline / Standard Range |
| :--- | :--- | :--- |
| **Glucose** | Plasma glucose concentration 2 hours in an oral glucose tolerance test | Normal: < 140 mg/dL |
| **Blood Pressure** | Diastolic blood pressure (mm Hg) | Normal: 60 - 80 mm Hg |
| **Skin Thickness** | Triceps skin fold thickness (mm) | Used to estimate subcutaneous body fat |
| **Insulin** | 2-Hour serum insulin (mu U/ml) | Varies, fasting level normally < 25 mu U/ml |
| **BMI** | Body mass index (weight in kg / (height in m)²) | Healthy range: 18.5 - 24.9 |
| **Diabetes Pedigree** | A function scoring diabetes genetic family history | Higher values indicate strong genetic predisposition |
| **Age** | Age of the patient (Years) | Restricted to patients $\ge$ 21 years old |

---

## 📊 Model Performance Summary

A comparison of multiple classifiers was executed during the training phase. The **Decision Tree Classifier** achieved the highest accuracy score and was selected as the dashboard's prediction engine:

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | **77.27** | **65.57** | **74.07** | **69.57** |
| KNN | 76.62 | 69.57 | 59.26 | 64.00 |
| Random Forest | 75.32 | 67.39 | 57.41 | 62.00 |
| SVM | 73.38 | 65.85 | 50.00 | 56.84 |
| Logistic Regression | 72.08 | 62.22 | 51.85 | 56.57 |

---

## 🖥️ Dashboard Layout & Features

1. **Patient Predictor Tab**:
   * Contains interactive input sliders for clinical variables.
   * Runs the StandardScaler scaling pipeline.
   * Feeds inputs to the Decision Tree model to display binary risk status ("Diabetic" vs "Non-Diabetic").
   * Displays the risk probability (confidence score) and lists diagnostic suggestions.

2. **Model Analytics Tab**:
   * Displays accuracy, precision, recall, and F1-score indicator cards.
   * Renders the overall performance table and the pre-computed comparison graphs.
   * Computes and graphs **Feature Importances** dynamically, showing which patient metrics (e.g., Glucose, BMI) contribute most heavily to predictions.

3. **Dataset Exploration Tab**:
   * Inspects baseline statistics (mean, standard deviation, bounds) of the Pima Indians dataset.
   * Displays interactive feature correlation heatmaps and pre-saved distribution plots.
