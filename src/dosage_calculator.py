import streamlit as st

def convert_to_mg(value, unit):
    unit = unit.lower()
    if unit in ['g', 'gram', 'grams']:
        return value * 1000
    elif unit in ['mcg', 'microgram', 'micrograms']:
        return value / 1000
    elif unit in ['mg', 'milligram', 'milligrams']:
        return value
    else:
        raise ValueError(f"Invalid unit: {unit}")

def convert_weight_to_kg(value, unit):
    unit = unit.lower()
    if unit in ['lb', 'pound', 'pounds']:
        return value * 0.453592
    elif unit in ['kg', 'kilogram', 'kilograms']:
        return value
    else:
        raise ValueError(f"Invalid weight unit: {unit}")

def dosage_calculator_ui():
    st.title("⚖️ Weight-Based Dosage Calculator")
    st.caption("Prototype only — not for clinical use.")

    weight_value = st.number_input("Patient Weight", min_value=0, step=1)
    weight_unit = st.selectbox("Weight Unit", ["kg", "lb"])

    dose_per_kg_value = st.number_input("Dosage per kg (daily)", min_value=0, step=1)
    dose_per_kg_unit = st.selectbox("Dosage Unit", ["mg", "mcg", "g"])

    frequency = st.selectbox("Frequency", [
        "Once per day", "Twice per day", "Three times/day", "Four times/day",
        "Every 4 hours", "Every 3 hours", "Every 2 hours", "Every hour"
    ])

    freq_map = {
        "Once per day": 1,
        "Twice per day": 2,
        "Three times/day": 3,
        "Four times/day": 4,
        "Every 4 hours": 6,
        "Every 3 hours": 8,
        "Every 2 hours": 12,
        "Every hour": 24
    }

    if weight_value > 0 and dose_per_kg_value > 0:
        weight_kg = convert_weight_to_kg(weight_value, weight_unit)
        daily_dose_mg = weight_kg * convert_to_mg(dose_per_kg_value, dose_per_kg_unit)
        single_dose_mg = daily_dose_mg / freq_map[frequency]

        st.success(
            f"💉 **Single Dose:** {single_dose_mg:.2f} mg\n"
            f"📅 **Total Daily Dose:** {daily_dose_mg:.2f} mg"
        )
