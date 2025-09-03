import streamlit as st
import json
import pandas as pd
import altair as alt
from pathlib import Path

def amr_checker_ui():
    st.set_page_config(page_title="AMR Resistance Checker", layout="wide")
    st.markdown("""
        <h1 style='text-align: center; color: #003366;'>🦠 AMR Resistance Checker</h1>
        <p style='text-align: center; font-size: 16px;'>Analyze bacterial resistance percentages for antibiotics.</p>
        <hr>
    """, unsafe_allow_html=True)

    # Load JSON data
    dataset2_path = Path("data/AMR-resistance-db.json")

    @st.cache_data
    def load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    data2 = load_json(dataset2_path)

    # --- Layout: two columns for selections ---
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Filters")
        bacteria_list = sorted(data2.keys())
        bacteria_selected = st.selectbox("Select Bacteria", [""] + bacteria_list)

        # Dynamic drug list
        if bacteria_selected:
            drugs_for_bact = sorted(data2[bacteria_selected].keys())
            drug_selected = st.selectbox("Select Drug", [""] + drugs_for_bact)
        else:
            all_drugs = sorted({drug for drugs in data2.values() for drug in drugs.keys()})
            drug_selected = st.selectbox("Select Drug", [""] + all_drugs)

    # --- Chart area ---
    with col2:
        def draw_chart(df, title, category_label):
            chart = alt.Chart(df).mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5
            ).encode(
                x=alt.X("Name", title=category_label, sort='-y'),
                y=alt.Y("Resistance (%)", title="Resistance (%)"),
                color=alt.Color("Resistance (%)", scale=alt.Scale(scheme='redyellowgreen')),
                tooltip=["Name", "Resistance (%)"]
            ).properties(
                title=title,
                width=600,
                height=400
            )
            st.altair_chart(chart, use_container_width=True)

        if bacteria_selected and not drug_selected:
            chart_data = [{"Name": drug, "Resistance (%)": val} 
                          for drug, val in data2[bacteria_selected].items()]
            df = pd.DataFrame(chart_data).sort_values("Resistance (%)", ascending=False)
            draw_chart(df, f"Resistance % for {bacteria_selected}", "Drug")

        elif drug_selected and not bacteria_selected:
            chart_data = [{"Name": bact, "Resistance (%)": drugs[drug_selected]} 
                          for bact, drugs in data2.items() if drug_selected in drugs]
            df = pd.DataFrame(chart_data).sort_values("Resistance (%)", ascending=False)
            draw_chart(df, f"Resistance % of {drug_selected} across bacteria", "Bacteria")

        elif bacteria_selected and drug_selected:
            # Show both charts
            chart_data1 = [{"Name": drug, "Resistance (%)": val} 
                           for drug, val in data2[bacteria_selected].items()]
            df1 = pd.DataFrame(chart_data1).sort_values("Resistance (%)", ascending=False)
            draw_chart(df1, f"Resistance % for {bacteria_selected}", "Drug")

            chart_data2 = [{"Name": bact, "Resistance (%)": drugs[drug_selected]} 
                           for bact, drugs in data2.items() if drug_selected in drugs]
            df2 = pd.DataFrame(chart_data2).sort_values("Resistance (%)", ascending=False)
            draw_chart(df2, f"Resistance % of {drug_selected} across bacteria", "Bacteria")

            # --- Add statement ---
            resistance_value = data2[bacteria_selected].get(drug_selected, None)
            if resistance_value is not None:
                st.markdown(f"**{bacteria_selected}** has **{resistance_value}%** resistance against **{drug_selected}**.")

        else:
            st.info("Select a Bacteria, a Drug, or both to see resistance charts.")

        # --- Reference ---
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Reference:** Pakistan AMR surveillance report 2022.", unsafe_allow_html=True)
