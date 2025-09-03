# src/drug_encyclopedia.py
import streamlit as st
import json
from pathlib import Path

def drug_encyclopedia_ui():
    st.title("📚 Drug Encyclopedia")
    st.markdown("Browse detailed drug information from the pharmacopedia database.\n---")

    # Path to your JSON file
    DATA_PATH = Path(__file__).parent.parent / "data" / "pharmacopedia.json"

    @st.cache_data
    def load_drug_data():
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
        return data

    drugs = load_drug_data()
    drug_names = [drug["Name"] for drug in drugs]

    # Searchable dropdown
    selected_drug_name = st.selectbox("Search or select a drug:", ["-- Select a drug --"] + drug_names)

    if selected_drug_name != "-- Select a drug --":
        # Find the selected drug
        drug = next((d for d in drugs if d["Name"] == selected_drug_name), None)

        if drug:
            # Display info in a nice layout
            st.markdown(f"## {drug['Name']}")
            st.markdown(f"**State:** {drug.get('State', 'N/A')}")

            # Two-column layout for some fields
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**General Function:** {drug.get('General Function', 'N/A')}")
                st.markdown(f"**Specific Function:** {drug.get('Specific Function', 'N/A')}")
                st.markdown(f"**Indication:** {drug.get('Indication', 'N/A')}")
                st.markdown(f"**Pharmacodynamics:** {drug.get('Pharmacodynamics', 'N/A')}")
                st.markdown(f"**Mechanism of action:** {drug.get('Mechanism of action', 'N/A')}")

            with col2:
                st.markdown(f"**Toxicity:** {drug.get('Toxicity', 'N/A')}")
                st.markdown(f"**Metabolism:** {drug.get('Metabolism', 'N/A')}")
                st.markdown(f"**Half life:** {drug.get('Half life', 'N/A')}")
                st.markdown(f"**Route of elimination:** {drug.get('Route of elimination', 'N/A')}")
                st.markdown(f"**Volume of distribution:** {drug.get('Volume of distribution', 'N/A')}")
                st.markdown(f"**Clearance:** {drug.get('Clearance', 'N/A')}")

            st.markdown("---")
            st.markdown("### Chemical Properties")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Melting Point:** {drug.get('Melting Point', 'N/A')}")
            with col2:
                if "Molecular weight" in drug:
                    st.markdown(f"**Molecular weight:** {drug.get('Molecular weight', 'N/A')}")
                if "Molecular formula" in drug:
                    st.markdown(f"**Molecular formula:** {drug.get('Molecular formula', 'N/A')}")

            st.markdown(f"**Description:** {drug.get('Description', 'N/A')}")
            st.markdown(f"**General reference:** {drug.get('General reference', 'N/A')}")
