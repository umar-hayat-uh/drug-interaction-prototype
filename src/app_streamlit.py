"""
Minimal Streamlit demo app.

Run:
    streamlit run src/app_streamlit.py
"""
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Drug Interaction Prototype", layout="centered")

st.title("🩺 Drug Interaction Prototype (demo)")
st.markdown("**Prototype only — not for clinical use.**")

data_dir = Path('data')
model_path = Path('models/xgb_model.joblib')
processed_csv = data_dir / "processed.csv"
original_csv = data_dir / "db_drug_interactions.csv"  # Your raw CSV with descriptions

st.sidebar.header("Data / Model")
st.sidebar.write("Place `processed.csv` and `db_drug_interactions.csv` in `data/`, model in `models/`")

@st.cache_data(show_spinner=True)
def load_label_encoder_and_severity_map():
    df = pd.read_csv(processed_csv)
    # Normalize drug names: lowercase & strip spaces
    drugs = pd.Series(list(df['Drug 1'].dropna()) + list(df['Drug 2'].dropna()))
    drugs = drugs.str.strip().str.lower()
    le_drug = LabelEncoder()
    le_drug.fit(drugs.astype(str))

    severity_labels = df['severity'].fillna('unknown').astype(str).unique()
    le_sev = LabelEncoder()
    le_sev.fit(severity_labels)
    severity_map = {code: label for code, label in enumerate(le_sev.classes_)}

    return le_drug, severity_map

@st.cache_data(show_spinner=True)
def load_interactions():
    df = pd.read_csv(original_csv)
    # Normalize drug columns for lookup
    df['Drug 1 norm'] = df['Drug 1'].str.strip().str.lower()
    df['Drug 2 norm'] = df['Drug 2'].str.strip().str.lower()
    return df

@st.cache_resource(show_spinner=True)
def load_model():
    return joblib.load(model_path)

def normalize_drug_name(name: str) -> str:
    return name.strip().lower()

le_drug, severity_map = load_label_encoder_and_severity_map()
model = load_model()
interactions_df = load_interactions()

drug1_input = st.text_input("Drug 1 (exact name)")
drug2_input = st.text_input("Drug 2 (exact name)")

if st.button("Predict (demo)"):
    if not model_path.exists():
        st.error("Model not found. Run `src/train_model.py` first and save to models/xgb_model.joblib")
    elif not processed_csv.exists():
        st.error("Processed data not found. Run preprocessing script first and save to data/processed.csv")
    elif not original_csv.exists():
        st.error("Original interactions CSV not found. Place `db_drug_interactions.csv` in data/")
    else:
        d1_norm = normalize_drug_name(drug1_input)
        d2_norm = normalize_drug_name(drug2_input)

        if d1_norm not in le_drug.classes_:
            st.error(f"Drug 1 '{drug1_input}' (normalized to '{d1_norm}') not found in known drugs.")
            st.stop()
        if d2_norm not in le_drug.classes_:
            st.error(f"Drug 2 '{drug2_input}' (normalized to '{d2_norm}') not found in known drugs.")
            st.stop()

        d1_enc = le_drug.transform([d1_norm])[0]
        d2_enc = le_drug.transform([d2_norm])[0]

        pred_proba = model.predict_proba([[d1_enc, d2_enc]])
        pred_label = pred_proba.argmax(axis=1)[0]
        severity_str = severity_map.get(pred_label, "unknown")

        # Look up interaction description (both orders)
        cond1 = (interactions_df['Drug 1 norm'] == d1_norm) & (interactions_df['Drug 2 norm'] == d2_norm)
        cond2 = (interactions_df['Drug 1 norm'] == d2_norm) & (interactions_df['Drug 2 norm'] == d1_norm)
        filtered = interactions_df[cond1 | cond2]

        if not filtered.empty:
            description = filtered.iloc[0]['Interaction Description']
        else:
            description = "No detailed interaction description found for this drug pair."

        st.success(f"Predicted severity: **{severity_str}** (class id: {pred_label})")
        st.markdown(f"### Interaction Description:")
        st.write(description)