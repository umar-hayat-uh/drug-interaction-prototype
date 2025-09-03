# src/dosage_calculator.py
import streamlit as st

def dosage_calculator_ui():
    st.set_page_config(page_title="⚖️ Weight-Based Dosage Calculator", layout="wide")

    st.title("⚖️ Weight-Based Dosage Calculator")
    st.caption("Prototype only — not for clinical use.")

    # -----------------------
    # Solid Form (Tablets)
    # -----------------------
    st.subheader("💊 Solid Dosage (Tablet/Capsule)")
    with st.form("solid_form"):
        weight = st.number_input("Patient Weight", min_value=1.0, value=25.0, step=0.5)
        dosage_per_kg = st.number_input("Dose per mg/kg", min_value=0.0, value=15.0, step=0.5)
        tablet_strength = st.number_input("Tablet strength (mg per tablet)", min_value=1.0, value=250.0, step=1.0)
        frequency = st.selectbox("Frequency", ["Once/day", "Twice/day", "Three times/day", "Four times/day"])
        solid_submit = st.form_submit_button("Calculate Solid Dose")

    if solid_submit:
        freq_map = {"Once/day": 1, "Twice/day": 2, "Three times/day": 3, "Four times/day": 4}
        total_daily_dose = weight * dosage_per_kg
        single_dose = total_daily_dose / freq_map[frequency]
        tablets_needed = single_dose / tablet_strength

        st.success(f"""
        📅 **Total Daily Dose**: {total_daily_dose:.2f} mg  
        💊 **Single Dose**: {single_dose:.2f} mg  
        🔢 **Tablets per dose**: {tablets_needed:.2f} tablets
        """)

    # -----------------------
    # Liquid Form (Syrup)
    # -----------------------
    st.subheader("🧴 Liquid Dosage (Syrup)")
    with st.form("liquid_form"):
        dosage_mg = st.number_input("Medicine Concentration (mg/ml)", min_value=0, step=1, value=0)
        concentration = st.number_input("Liquid Dose (ml)", min_value=0, step=1, value=0)

        # Calculate the result
        calculated_dose = dosage_mg * concentration

        # Show the result in an input field
        answer = st.text_input("💉 Result (ml)", value=f"{calculated_dose:.2f}", disabled=True)

        liquid_submit = st.form_submit_button("Calculate Liquid Dose")