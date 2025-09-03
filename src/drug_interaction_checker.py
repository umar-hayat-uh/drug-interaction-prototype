import streamlit as st
import json
import re
from pathlib import Path

def drug_interaction_checker_ui():
    json_path = Path("data/drug-interactions.json")

    @st.cache_data
    def load_interactions_json():
        with open(json_path, "r") as f:
            data = json.load(f)
        return data["drug_interactions"]

    drug_interactions = load_interactions_json()

    # Collect all unique drugs
    all_drugs = set()
    for severity in ["major", "moderate", "minor"]:
        for item in drug_interactions.get(severity, []):
            all_drugs.add(item["drug_a"])
            all_drugs.add(item["drug_b"])

    st.subheader("💊 Drug Interaction Checker")
    col1, col2 = st.columns(2)
    with col1:
        drug1 = st.selectbox("Select Drug 1", sorted(all_drugs))
    with col2:
        interacting_drugs = set()
        for severity in ["major", "moderate", "minor"]:
            for item in drug_interactions.get(severity, []):
                if item["drug_a"] == drug1:
                    interacting_drugs.add(item["drug_b"])
                elif item["drug_b"] == drug1:
                    interacting_drugs.add(item["drug_a"])
        drug2 = st.selectbox("Select Drug 2", sorted(interacting_drugs))

    # Disclaimer
    st.markdown(
        """
        <div style="
            border:2px solid #f28b82;
            background-color:#fdecea;
            padding:12px;
            border-radius:10px;
            margin:15px 0;
            color:#b71c1c;
            font-size:15px;
        ">
        ⚠️ <strong>Warning:</strong> If no interactions are found between two drugs, it does not necessarily mean that no interactions exist. Always consult with a healthcare professional.
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Check Interaction"):
        if not drug1 or not drug2:
            st.warning("⚠️ Please select both drugs.")
            return

        # Find the interaction
        interaction = None
        for severity in ["major", "moderate", "minor"]:
            for item in drug_interactions.get(severity, []):
                if (item["drug_a"] == drug1 and item["drug_b"] == drug2) or \
                   (item["drug_b"] == drug1 and item["drug_a"] == drug2):
                    interaction = item
                    break
            if interaction:
                break

        if interaction:
            severity = interaction['severity'].lower()
            color_map = {
                "major": "#ff6961",      # red
                "moderate": "#ffb347",   # orange
                "minor": "#fef68a"       # yellow
            }
            bg_color = color_map.get(severity, "#d3d3d3")

            # Clean text only for minor interactions
            def clean_text(value):
                if not isinstance(value, str):
                    return value
                value = re.sub(r"</?div.*?>", "", value, flags=re.IGNORECASE)
                value = re.sub(r"<.*?>", "", value)
                return value.strip()

            mechanism = interaction.get('mechanism', 'N/A')
            effect = interaction.get('effect', 'N/A')
            safer_alt = interaction.get('Safer_alternative')
            rationale = interaction.get('rationale', 'N/A')
            reference = interaction.get('reference', 'N/A')

            if severity == "minor":
                mechanism = clean_text(mechanism)
                effect = clean_text(effect)
                safer_alt = clean_text(safer_alt) if safer_alt else None
                rationale = clean_text(rationale)
                reference = clean_text(reference)

            # Build HTML
            details_html = f"<p><strong>Mechanism:</strong> {mechanism}</p>"
            details_html += f"<p><strong>Effect:</strong> {effect}</p>"
            if severity in ["major", "moderate"] and safer_alt:
                details_html += f"<p><strong>Safer Alternative:</strong> {safer_alt}</p>"
            details_html += f"<p><strong>Rationale:</strong> {rationale}</p>"
            details_html += f"<p><strong>Reference:</strong> {reference}</p>"

            st.markdown(
                f"""
                <div style="
                    border-radius: 12px;
                    padding: 20px;
                    background-color: #f9f9f9;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                ">
                    <h2 style="margin-top:0;">💊 {drug1} ↔ {drug2}</h2>
                    <div style="
                        display:flex;
                        align-items:center;
                        gap:10px;
                        margin-bottom:15px;
                    ">
                        <span style="
                            padding:5px 10px;
                            border-radius:5px;
                            background-color:{bg_color};
                            font-weight:bold;
                            color:#000;
                        ">{interaction['severity'].upper()}</span>
                    </div>
                    {details_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("✅ No known interaction found between these drugs.")
