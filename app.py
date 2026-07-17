import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configurations
st.set_page_config(
    page_title="Diabetes Risk Predictor & Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'All Docs', 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'All Docs', 'models', 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'All Docs', 'models', 'feature_names.pkl')
RESULTS_CSV = os.path.join(BASE_DIR, 'All Docs', 'files', 'model_comparison_results.csv')
RESULT_IMGS_DIR = os.path.join(BASE_DIR, 'All Docs', 'result')

# Helper functions to load models safely
@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_PATH)
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None

model, scaler, feature_names = load_ml_assets()

# Custom CSS for Premium Design & Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* General styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Styles */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk card styles */
    .risk-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
        border-radius: 12px;
        padding: 25px;
        color: #991b1b;
        box-shadow: 0 4px 6px -1px rgba(153, 27, 27, 0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 25px;
        color: #166534;
        box-shadow: 0 4px 6px -1px rgba(22, 101, 52, 0.1);
    }
    
    /* Heading Style */
    .main-title {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #475569;
        font-size: 1.1rem;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Styling Streamlit Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.2);
        transition: opacity 0.2s, transform 0.2s;
    }
    div.stButton > button:first-child:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.3);
    }
    div.stButton > button:first-child:active {
        transform: translateY(0px);
    }
</style>
""", unsafe_allow_html=True)

# Main Title Section
st.markdown('<div class="main-title">🩺 Diabetes Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">An interactive portal utilizing a Decision Tree model to assess diabetes risks and analyze clinical metrics.</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.markdown("### **Navigation & Info**")
    st.write("This dashboard provides clinical evaluations based on the Pima Indians Diabetes Dataset.")
    st.markdown("---")
    
    # Display Active Model Details
    st.markdown("### **Active Model**")
    st.success("🏆 **Decision Tree Classifier**")
    
    # Sidebar Metrics
    st.markdown("#### **Evaluation Metrics**")
    st.metric(label="Accuracy Score", value="77.27%")
    st.metric(label="Recall Score", value="74.07%")
    st.metric(label="Precision Score", value="65.57%")
    
    st.markdown("---")
    st.info("💡 *Note: The model excludes 'Pregnancies' as a feature, focusing purely on metabolic and lifestyle indicators.*")

# Tabs definition
tab_predictor, tab_insights, tab_data = st.tabs([
    "🩺 Patient Predictor", 
    "📊 Model Analytics", 
    "📈 Dataset Exploration"
])

# --- TAB 1: PREDICTOR ---
with tab_predictor:
    st.write("Fill in the patient's clinical variables below to predict their diabetes risk category.")
    
    col_input, col_output = st.columns([3, 2], gap="large")
    
    with col_input:
        st.subheader("📋 Patient Clinical Profiles")
        
        with st.form("prediction_form"):
            # Custom input layout
            col_inner1, col_inner2 = st.columns(2)
            
            with col_inner1:
                glucose = st.slider(
                    "Glucose (mg/dL)", 
                    min_value=0, max_value=200, value=120, 
                    help="Plasma glucose concentration 2 hours in an oral glucose tolerance test. Normal: < 140 mg/dL"
                )
                blood_pressure = st.slider(
                    "Blood Pressure (mm Hg)", 
                    min_value=0, max_value=122, value=70,
                    help="Diastolic blood pressure. Normal: 60-80 mm Hg"
                )
                skin_thickness = st.slider(
                    "Skin Thickness (mm)", 
                    min_value=0, max_value=99, value=20,
                    help="Triceps skin fold thickness. Used to estimate body fat."
                )
                insulin = st.slider(
                    "Insulin (mu U/ml)", 
                    min_value=0, max_value=846, value=80,
                    help="2-Hour serum insulin levels."
                )
            
            with col_inner2:
                bmi = st.slider(
                    "Body Mass Index (BMI)", 
                    min_value=0.0, max_value=67.1, value=32.0, step=0.1,
                    help="Weight in kg / (height in m)^2. Healthy range: 18.5 - 24.9"
                )
                dpf = st.slider(
                    "Diabetes Pedigree Function", 
                    min_value=0.078, max_value=2.42, value=0.37, step=0.001,
                    help="A score representing family history and genetic predisposition to diabetes."
                )
                age = st.slider(
                    "Age (Years)", 
                    min_value=21, max_value=81, value=33,
                    help="Age of the patient."
                )
            
            # Predict Button
            submit_btn = st.form_submit_button("Predict Diabetes Risk")
            
    with col_output:
        st.subheader("🎯 Prediction Output")
        
        if submit_btn and model is not None and scaler is not None:
            # Prepare scaled features in correct shape
            # Order: ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            input_df = pd.DataFrame([[glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], columns=feature_names)
            scaled_features = scaler.transform(input_df)
            
            # Prediction
            pred = model.predict(scaled_features)[0]
            
            # Probability calculation (if supported by active model)
            try:
                proba = model.predict_proba(scaled_features)[0]
                diabetic_proba = proba[1] * 100
            except:
                # Fallback if model doesn't support predict_proba
                diabetic_proba = 100.0 if pred == 1 else 0.0
                
            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ High Risk Detected</h3>
                    <p>The patient is classified as <strong>Diabetic</strong>.</p>
                    <hr style="border-color: #fca5a5;">
                    <p style="font-size: 1.1rem; margin-bottom: 5px;"><strong>Model Confidence Score:</strong></p>
                    <h2 style="margin: 0; color: #991b1b;">{diabetic_proba:.1f}% Risk</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("🩺 **Clinical Recommendations:**")
                st.markdown("""
                *   **Further Screening**: Schedule a fasting plasma glucose (FPG) test or HbA1c test to confirm results.
                *   **Dietary Intervention**: Suggest lowering carbohydrate and sugar intake, transitioning to high-fiber foods.
                *   **Activity Plan**: Recommend at least 150 minutes of moderate physical activity per week.
                *   **Metabolic Review**: Since Glucose/BMI are key drivers, closely monitor glycemic responses and target BMI reduction.
                """)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ Low Risk Detected</h3>
                    <p>The patient is classified as <strong>Non-Diabetic</strong>.</p>
                    <hr style="border-color: #86efac;">
                    <p style="font-size: 1.1rem; margin-bottom: 5px;"><strong>Model Confidence Score:</strong></p>
                    <h2 style="margin: 0; color: #166534;">{diabetic_proba:.1f}% Risk</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("🥗 **Preventative Recommendations:**")
                st.markdown("""
                *   **Maintain Healthy Habits**: Continue eating balanced meals and engaging in regular exercise.
                *   **Routine Screenings**: Recommend checking blood glucose levels annually, especially for patients over 45.
                *   **Monitor BMI**: Encourage weight control strategies to keep BMI in the normal range (18.5 - 24.9).
                """)
        else:
            st.info("ℹ️ Enter the patient's clinical attributes on the left and click **Predict Diabetes Risk** to view results here.")
            
            # Simple card to show clinical baseline ranges
            st.markdown("""
            <div style="background-color: #f1f5f9; border-radius: 12px; padding: 20px; border: 1px solid #e2e8f0; margin-top: 15px;">
                <h4 style="margin-top:0;">💡 Standard Normal Ranges Reference</h4>
                <ul style="margin-bottom:0;">
                    <li><strong>Glucose</strong>: < 140 mg/dL (fasting/post-meal ranges differ)</li>
                    <li><strong>Blood Pressure</strong>: Diastolic between 60 - 80 mm Hg</li>
                    <li><strong>BMI</strong>: Normal weight is 18.5 - 24.9</li>
                    <li><strong>Insulin</strong>: Fasting < 25 mu U/ml, post-meal varies</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


# --- TAB 2: MODEL ANALYTICS ---
with tab_insights:
    st.subheader("📊 Classifier Performance Analysis")
    st.write("Compare the performance parameters of the 5 machine learning models evaluated during training.")
    
    # 4 metrics cards at the top
    col_acc, col_rec, col_pr, col_f1 = st.columns(4)
    with col_acc:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-val">77.27%</div>
            <div style="color: #0d9488; font-size: 0.85rem; font-weight: 500;">Best Performing</div>
        </div>
        """, unsafe_allow_html=True)
    with col_rec:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Recall</div>
            <div class="metric-val">74.07%</div>
            <div style="color: #0d9488; font-size: 0.85rem; font-weight: 500;">High Sensitivity</div>
        </div>
        """, unsafe_allow_html=True)
    with col_pr:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Precision</div>
            <div class="metric-val">65.57%</div>
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 500;">Target Positive Rate</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">F1-Score</div>
            <div class="metric-val">69.57%</div>
            <div style="color: #64748b; font-size: 0.85rem; font-weight: 500;">Balanced Score</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_tbl, col_fi = st.columns([3, 2], gap="large")
    
    with col_tbl:
        st.markdown("### 📋 Model Comparison Matrix")
        # Load results csv
        if os.path.exists(RESULTS_CSV):
            results_df = pd.read_csv(RESULTS_CSV)
            st.dataframe(results_df.style.highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-score'], color='#ccfbf1'), use_container_width=True)
        else:
            st.info("Model comparison results CSV not found.")
            
        # Display saved plot
        acc_comp_img = os.path.join(RESULT_IMGS_DIR, 'fig4_accuracy_comparison.png')
        if os.path.exists(acc_comp_img):
            st.image(acc_comp_img, caption="Figure 4: Accuracy Comparison across Models", use_container_width=True)
            
    with col_fi:
        st.markdown("### ⚡ Feature Importances")
        st.write("Relative importance of clinical variables computed dynamically from the Decision Tree model:")
        
        if model is not None and feature_names is not None:
            # Dynamic Feature Importance
            try:
                importances = model.feature_importances_
                fi_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot Seaborn/Matplotlib
                fig, ax = plt.subplots(figsize=(6, 4.5))
                sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis', ax=ax)
                ax.set_title("Relative Feature Importance (Decision Tree)")
                ax.set_xlabel("Importance Value")
                ax.set_ylabel("")
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.write(f"Could not compute feature importance: {e}")
        
        # Display confusion matrix image
        cm_img = os.path.join(RESULT_IMGS_DIR, 'fig7_confusion_matrix.png')
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Figure 7: Confusion Matrix (Decision Tree)", use_container_width=True)

# --- TAB 3: DATASET EXPLORATION ---
with tab_data:
    st.subheader("📈 Dataset Exploration & Visualizations")
    st.write("Analyze structural attributes, correlations, and sample distributions from the baseline Pima dataset.")
    
    # Load and show stats
    pima_csv = os.path.join(BASE_DIR, 'All Docs', 'files', 'pima.csv')
    if os.path.exists(pima_csv):
        # We need headers for Pima
        pima_cols = ['Nausea', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(pima_csv, names=pima_cols)
        
        col_summary, col_dist = st.columns([2, 3], gap="large")
        
        with col_summary:
            st.markdown("### 📊 Dataset Overview")
            st.dataframe(df.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True)
            st.write("Total Dataset Size:", df.shape[0], "records")
            st.write("Diabetic Patients (Outcome=1):", df[df['Outcome'] == 1].shape[0], f"({(df[df['Outcome'] == 1].shape[0] / df.shape[0] * 100):.1f}%)")
            st.write("Healthy Patients (Outcome=0):", df[df['Outcome'] == 0].shape[0], f"({(df[df['Outcome'] == 0].shape[0] / df.shape[0] * 100):.1f}%)")
            
        with col_dist:
            st.markdown("### 🗺️ Correlation Matrix")
            # Create a simple heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title("Feature Correlations (Pima Dataset)")
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        
        # Display other saved distributions
        st.markdown("### 🖼️ Saved Data Distributions from Notebooks")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            dist_img = os.path.join(RESULT_IMGS_DIR, 'fig2_feature_distributions.png')
            if os.path.exists(dist_img):
                st.image(dist_img, caption="Figure 2: Clinical Feature Distributions", use_container_width=True)
        with col_img2:
            class_dist_img = os.path.join(RESULT_IMGS_DIR, 'fig1_class_distribution.png')
            if os.path.exists(class_dist_img):
                st.image(class_dist_img, caption="Figure 1: Outcome Class Distribution", use_container_width=True)
    else:
        st.info("Pima dataset file not found in 'All Docs/files/pima.csv'.")
