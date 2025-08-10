"""
Minimal Streamlit demo app.

Run:
    streamlit run src/app_streamlit.py
"""
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Drug Interaction Prototype", layout="centered")

st.title("🩺 Drug Interaction Prototype (demo)")
st.markdown("**Prototype only — not for clinical use.**")

data_dir = Path('data')
model_path = Path('models/xgb_model.joblib')

st.sidebar.header("Data / Model")
st.sidebar.write("Place `processed.csv` in `data/` and trained model in `models/`")

drug1 = st.text_input("Drug 1 (exact name or encoded id)")
drug2 = st.text_input("Drug 2 (exact name or encoded id)")

if st.button("Predict (demo)"):
    if not model_path.exists():
        st.error("Model not found. Run `src/train_model.py` first and save to models/xgb_model.joblib")
    else:
        model = joblib.load(model_path)
        # This demo expects encoded integers; in a full app you'd map names -> enc ids
        try:
            d1 = int(drug1)
            d2 = int(drug2)
            pred = model.predict([[d1,d2]])
            st.success(f"Predicted severity class id: {int(pred[0])}")
        except Exception as e:
            st.error(f"Input error (demo expects encoded integer ids). Full app should map names -> enc ids. Error: {e}")
