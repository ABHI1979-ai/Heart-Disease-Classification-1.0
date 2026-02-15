import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, matthews_corrcoef

st.set_page_config(page_title="Heart Disease Predictor")
st.title("Heart Disease Classification Dashboard")

# Features for the UCI Heart Disease dataset
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

uploaded_file = st.file_uploader("Upload Test Data (CSV)", type="csv")

model_name = st.selectbox("Select Model", 
    ["Logistic_Regression", "Decision_Tree", "kNN", "Naive_Bayes", "Random_Forest", "XGBoost"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    
    if 'num' in df_test.columns:
        df_test = df_test.fillna(0)
        for col in df_test.select_dtypes(include=['object']).columns:
            df_test[col] = df_test[col].astype('category').cat.codes
        
        X_test = df_test[FEATURES]
        y_test = (df_test['num'] > 0).astype(int)
        
        with open(f"model/{model_name}.pkl", 'rb') as f:
            model = pickle.load(f)
            
        if model_name in ["Logistic_Regression", "kNN"]:
            with open("model/scaler.pkl", 'rb') as f:
                scaler = pickle.load(f)
            X_test = scaler.transform(X_test)
            
        y_pred = model.predict(X_test)
        
        st.subheader(f"Results for {model_name.replace('_', ' ')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        c2.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
        c3.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)
    else:
        st.error("Error: The uploaded CSV must contain a 'num' column.")
