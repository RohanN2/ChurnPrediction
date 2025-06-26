import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from datetime import datetime
import os

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("Upload a customer dataset to predict churn, analyze risk tiers, and get actionable insights.")

# ------------------ Required Files Check ------------------
required_files = ['label_encoders.pkl', 'scaler.pkl', 'xgb_model.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing required files: {', '.join(missing_files)}. Please run the training script first.")
    st.stop()

# ------------------ Load Artifacts ------------------
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('xgb_model.pkl')

# List of categorical columns to encode
label_enc_cols = ['Gender', 'Location', 'Occupation', 'Income Bracket',
                  'Channel', 'Device Type', 'OS Version', 'Current Plan',
                  'Plan History', 'Payment Mode', 'Survey Feedback']

# ------------------ Preprocessing Function ------------------
def preprocess_input(df):
    df = df.copy()

    # Impute numeric and categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode categorical variables
    for col in label_enc_cols:
        if col in df.columns:
            le = label_encoders[col]
            df[col] = le.transform(df[col].astype(str))

    # Create tenure feature
    if 'Onboarding Date' in df.columns:
        df['Onboarding Date'] = pd.to_datetime(df['Onboarding Date'], errors='coerce')
        tenure_days = (pd.to_datetime("2025-06-01") - df['Onboarding Date']).dt.days
        df['Customer Tenure (months)'] = (tenure_days / 30.44).fillna(tenure_days.mean() / 30.44)

    # Drop irrelevant columns
    df = df.drop(columns=['CustomerID', 'Onboarding Date', 'Churned'], errors='ignore')

    return df

# ------------------ SHAP Visualization Helper (Static Plot Only) ------------------
def show_shap_plot(explainer, shap_values, X, index):
    """Display a clean and compact SHAP waterfall plot for a single instance"""
    st.markdown("#### 📊 Feature Impact on Churn Prediction")

    # Handle Explanation object or convert raw values
    if isinstance(shap_values, shap.Explanation):
        explanation = shap_values[index]
    else:
        explanation = shap.Explanation(
            values=shap_values[index],
            base_values=explainer.expected_value,
            data=X.iloc[index].values,
            feature_names=X.columns.tolist()
        )

    # Plot with reduced size
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    ax.set_title("Feature Contributions", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

# ------------------ Upload Section ------------------
uploaded_file = st.file_uploader("📤 Upload your CSV file with customer data", type=["csv"])

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Preprocess
        processed_df = preprocess_input(df_upload.copy())

        # Ensure all model features are present
        missing_features = set(model.feature_names_in_) - set(processed_df.columns)
        if missing_features:
            st.warning(f"⚠️ Missing expected features: {missing_features}")
            st.stop()

        # Scale features
        X = processed_df[model.feature_names_in_]
        X_scaled = scaler.transform(X)

        # Predict churn probability
        churn_prob = model.predict_proba(X_scaled)[:, 1]

        # Risk tier assignment
        def assign_risk_tier(prob):
            if prob < 0.4:
                return "Low"
            elif 0.4 <= prob < 0.7:
                return "Medium"
            else:
                return "High"

        risk_tiers = [assign_risk_tier(p) for p in churn_prob]

        # Generate recommendations
        def generate_recommendation(row):
            tier = row['Risk_Tier']
            features = row['Top_Features']

            actions = []

            if tier == 'Low':
                actions.append("Send monthly loyalty points")
            elif tier == 'Medium':
                actions.append("Offer $5 cashback for next renewal")
            elif tier == 'High':
                actions.append("Immediate support call + plan upgrade offer")

            if 'Tickets Raised' in features:
                actions.append("Escalate support ticket")
            if 'App Logins' in features:
                actions.append("Send re-engagement email + free month trial")
            if 'Payment Mode_Manual' in features:
                actions.append("Encourage auto-payment setup")
            if 'Credit Score' in features:
                actions.append("Offer credit-building tools")
            if 'Plan Downgrade' in features:
                actions.append("Suggest plan upgrade with bonus benefits")

            return list(set(actions))  # Remove duplicates

        # Explain with SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Get top contributing features per row
        shap_df = pd.DataFrame(shap_values, columns=X.columns)
        top_features = shap_df.apply(lambda row: row.sort_values(ascending=False).head(3).index.tolist(), axis=1)

        # Assemble final results
        result_df = df_upload.copy()
        result_df['Churn_Probability'] = churn_prob
        result_df['Risk_Tier'] = risk_tiers
        result_df['Top_Features'] = top_features
        result_df['Recommendations'] = result_df.apply(generate_recommendation, axis=1)

        # Display full DataFrame
        st.subheader("📊 Full Dataset with Predictions")
        st.dataframe(result_df.style.format({'Churn_Probability': '{:.2%}'}))

        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Results",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv'
        )

        # Optional: Show SHAP waterfall plot for one customer
        st.subheader("🔍 Model Explanation for Selected Customer")
        customer_index = st.slider("Select Customer Index", 0, len(result_df) - 1, 0)
        show_shap_plot(explainer, shap_values, X, customer_index)

        # Optional: Global feature importance
        st.subheader("🧩 Global Feature Importance")
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, plot_type="bar")
        st.pyplot(plt.gcf())
        plt.close()

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")