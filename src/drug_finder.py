import streamlit as st
import pandas as pd
from pathlib import Path

def drug_finder_ui():
    st.title("🔍 Drug Finder")
    st.markdown("Search drugs by **Name, Generic, or Class**")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "DrugData.csv"
    df = pd.read_csv(data_path)

    # Keep only required fields: Drug Name → Manufacturer
    selected_columns = [
        'Drug Name',
        'Generic Name',
        'Drug Class',
        'Indications',
        'Dosage Form',
        'Strength',
        'Route of Administration',
        'Mechanism of Action',
        'Side Effects',
        'Contraindications',
        'Interactions',
        'Warnings and Precautions',
        'Pregnancy Category',
        'Storage Conditions',
        'Manufacturer'
    ]
    df = df[selected_columns]

    # Search input
    query = st.text_input("Enter drug name, generic, or class:")
    view_mode = st.radio("View mode:", ["Table", "Vertical (Mobile-friendly)"], horizontal=True)

    if query:
        mask = (
            df['Drug Name'].str.contains(query, case=False, na=False) |
            df['Generic Name'].str.contains(query, case=False, na=False) |
            df['Drug Class'].str.contains(query, case=False, na=False)
        )
        results = df[mask]

        if not results.empty:
            if view_mode == "Table":
                st.dataframe(results, use_container_width=True)
            else:
                for _, row in results.iterrows():
                    with st.expander(f"{row['Drug Name']} ({row['Generic Name']})"):
                        for col in selected_columns:
                            st.markdown(f"**{col}:** {row[col]}")
        else:
            st.warning("No results found.")
