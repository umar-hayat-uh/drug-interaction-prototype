# src/drug_encyclopedia.py
import streamlit as st
import pandas as pd
import os

def drug_encyclopedia_ui():
    DATA_PATH = os.path.join("data", "drugbank_full.csv")

    st.markdown("""
        <style>
        .card {
            background-color: #f9f9f9;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 12px;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.12);
            word-wrap: break-word;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 4px 4px 12px rgba(0,0,0,0.15);
        }
        .card h4 {
            margin-top: 0;
            margin-bottom: 6px;
            color: #333333;
            font-size: 16px;
        }
        .card p {
            color: #000000;
            font-size: 14px;
        }
        .drug-title {
            background-color: #e0f7fa;
            padding: 12px;
            border-radius: 10px;
            font-size: 22px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }
        hr.section-divider {
            border: 0;
            border-top: 1px solid #ddd;
            margin: 20px 0;
        }
        @media only screen and (max-width: 768px) {
            .card { padding: 10px; }
            .card h4 { font-size: 14px; }
            .card p { font-size: 13px; }
            .drug-title { font-size: 18px; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Load CSV
    try:
        df = pd.read_csv(DATA_PATH, low_memory=False)
    except FileNotFoundError:
        st.error("❌ DrugBank CSV not found. Please place 'drugbank_full.csv' in the data/ folder.")
        return

    st.subheader("📚 Drug Encyclopedia")
    search_term = st.text_input("🔍 Enter drug name:")

    drug = None

    if search_term.strip():
        name_col = "name"  # exact column in CSV
        exact_match = df[df[name_col].str.lower() == search_term.lower()]
        if not exact_match.empty:
            drug = exact_match.iloc[0]
        else:
            partial_matches = df[df[name_col].str.contains(search_term, case=False, na=False)]
            if not partial_matches.empty:
                selected_name = st.selectbox(
                    "No exact match found. Please select a drug from the options:",
                    partial_matches[name_col].tolist()
                )
                drug = partial_matches[partial_matches[name_col] == selected_name].iloc[0]
            else:
                st.warning("No drugs found with that name or similar names.")

    if drug is not None:
        selected_name = drug["name"]
        st.markdown(f"<div class='drug-title'>💊 {selected_name}</div>", unsafe_allow_html=True)
        new_name = st.text_input("Drug Name", selected_name)

        # Map display fields to CSV columns
        basic_fields = {
            "State": "state",
            "Half Life": "half-life",
            "Route of Elimination": "route-of-elimination",
            "Volume of Distribution": "volume-of-distribution",
            "Clearance": "clearance",
            "Molecular Weight": "molecular-weight",
            "Molecular Formula": "molecular-formula",
            "Melting Point": "melting-point",
            "General Reference": "general-reference"
        }

        detailed_fields = {
            "Description": "description",
            "Indication": "indication",
            "Pharmacodynamics": "pharmacodynamics",
            "Mechanism of Action": "mechanism-of-action",
            "Toxicity": "toxicity",
            "Metabolism": "metabolism",
            "General Function": "targets_target_polypeptide_general-function",
            "Specific Function": "targets_target_polypeptide_specific-function"
        }

        classification_fields = {
            "Direct Parent": "classification_direct-parent",
            "Kingdom": "classification_kingdom",
            "Superclass": "classification_superclass",
            "Class": "classification_class",
            "Subclass": "classification_subclass"
        }

        def create_card(title, value):
            value_display = "No data found" if value is None or pd.isna(value) or str(value).strip() == "" else str(value)
            st.markdown(f"""
                <div class="card">
                    <h4>{title}</h4>
                    <p>{value_display}</p>
                </div>
            """, unsafe_allow_html=True)

        # Basic Info
        st.markdown("### 🧾 Basic Information")
        cols = st.columns(3)
        for i, (title, col_name) in enumerate(basic_fields.items()):
            value = drug[col_name] if col_name in drug.index else None
            with cols[i % 3]:
                create_card(title, value)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Detailed Info
        with st.expander("📖 Detailed Information", expanded=False):
            cols = st.columns(2)
            for i, (title, col_name) in enumerate(detailed_fields.items()):
                value = drug[col_name] if col_name in drug.index else None
                with cols[i % 2]:
                    create_card(title, value)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # Classification
        with st.expander("🏷️ Classification", expanded=False):
            cols = st.columns(3)
            for i, (title, col_name) in enumerate(classification_fields.items()):
                value = drug[col_name] if col_name in drug.index else None
                with cols[i % 3]:
                    create_card(title, value)

        if new_name != selected_name:
            st.success(f"✅ Name updated to: {new_name}")
