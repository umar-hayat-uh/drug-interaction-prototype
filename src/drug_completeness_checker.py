import streamlit as st
import pandas as pd
import os

def drug_completeness_checker_ui():
    DATA_PATH = os.path.join("data", "drugbank_full.csv")

    st.set_page_config(page_title="💎 Drug Completeness Checker", layout="wide")

    st.title("💎 Drug Completeness Checker")
    st.markdown("""
    This tool shows which drugs in the dataset have the **most complete data**.
    """)

    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except FileNotFoundError:
        st.error("❌ DrugBank CSV not found. Please place 'drugbank_full.csv' in the data/ folder.")
        return

    # Calculate completeness
    df['completeness'] = df.notna().sum(axis=1) / df.shape[1] * 100

    top_drugs_count = st.slider("Number of top drugs to show", min_value=5, max_value=50, value=10, step=5)
    top_drugs = df.sort_values(by='completeness', ascending=False).head(top_drugs_count)

    for _, row in top_drugs.iterrows():
        st.markdown(f"💊 **{row['name']}** — {row['completeness']:.1f}% data available")
