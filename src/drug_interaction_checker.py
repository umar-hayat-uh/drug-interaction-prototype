import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def drug_interaction_checker_ui():
    data_dir = Path('data')
    model_path = Path('models/xgb_model.joblib')
    processed_csv = data_dir / "processed.csv"
    original_csv = data_dir / "db_drug_interactions.csv"

    @st.cache_data
    def load_label_encoder_and_severity_map():
        df = pd.read_csv(processed_csv)
        drugs = pd.Series(list(df['Drug 1']) + list(df['Drug 2'])).str.strip().str.lower()
        le_drug = LabelEncoder()
        le_drug.fit(drugs.astype(str))

        severity_labels = df['severity'].fillna('unknown').astype(str).unique()
        le_sev = LabelEncoder()
        le_sev.fit(severity_labels)
        severity_map = {code: label for code, label in enumerate(le_sev.classes_)}
        return le_drug, severity_map

    @st.cache_data
    def load_interactions():
        df = pd.read_csv(original_csv)
        df['Drug 1 norm'] = df['Drug 1'].str.strip().str.lower()
        df['Drug 2 norm'] = df['Drug 2'].str.strip().str.lower()
        return df

    @st.cache_resource
    def load_model():
        return joblib.load(model_path)

    le_drug, severity_map = load_label_encoder_and_severity_map()
    model = load_model()
    interactions_df = load_interactions()

    st.subheader("Check Drug Interaction")
    col1, col2 = st.columns(2)
    with col1:
        drug1_input = st.text_input("Drug 1")
    with col2:
        drug2_input = st.text_input("Drug 2")

    if st.button("Check Interaction"):
        if not drug1_input or not drug2_input:
            st.warning("Please enter both drug names.")
            return

        d1_norm = drug1_input.strip().lower()
        d2_norm = drug2_input.strip().lower()

        if d1_norm not in le_drug.classes_ or d2_norm not in le_drug.classes_:
            st.error("One or both drugs not found in database.")
            return

        d1_enc = le_drug.transform([d1_norm])[0] # type: ignore
        d2_enc = le_drug.transform([d2_norm])[0] # type: ignore

        pred_proba = model.predict_proba([[d1_enc, d2_enc]])
        pred_label = pred_proba.argmax(axis=1)[0]
        severity_str = severity_map.get(pred_label, "unknown")

        cond1 = (interactions_df['Drug 1 norm'] == d1_norm) & (interactions_df['Drug 2 norm'] == d2_norm)
        cond2 = (interactions_df['Drug 1 norm'] == d2_norm) & (interactions_df['Drug 2 norm'] == d1_norm)
        filtered = interactions_df[cond1 | cond2]

        description = filtered.iloc[0]['Interaction Description'] if not filtered.empty else "No detailed description found."

        st.markdown(f"**Predicted severity:** `{severity_str.upper()}`")
        st.info(description)