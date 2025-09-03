"""
Minimal Streamlit demo app.

Run:
    streamlit run src/app_streamlit.py
"""
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="💊 Drug Tools Suite", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "🏠 Home",
        "⚠️ Drug Interaction Checker",
        "📏 Dosage Calculator",
        "📚 Drug Encyclopedia",
        "💎 Completeness Checker",
        "🔍 Drug Finder",
        "🦠 AMR Checker",
        "🧠 Pharm AI"
    ]
)

# Home Page
if page == "🏠 Home":
    st.title("💊 Drug Tools Suite")
    st.markdown("""
    Welcome to the **Drug Tools Suite** — a **prototype only** for educational use.
    
    **Features:**
    - ⚠️ **Drug Interaction Checker** — Check potential interactions between two drugs.
    - 📏 **Dosage Calculator** — Calculate medication doses for tablet or liquid forms.
    - 🧾 **Drug Description Finder** — Get AI-powered drug descriptions.

    ---
    **Disclaimer:** Not for clinical use. Always consult a licensed healthcare provider.
    """)

elif page == "⚠️ Drug Interaction Checker":
    from drug_interaction_checker import drug_interaction_checker_ui
    drug_interaction_checker_ui()

elif page == "📏 Dosage Calculator":
    from dosage_calculator import dosage_calculator_ui
    dosage_calculator_ui()
    
elif page == "📚 Pharmacopedia":
    from drug_encyclopedia import drug_encyclopedia_ui
    drug_encyclopedia_ui()

elif page == "💎 Completeness Checker":
    from drug_completeness_checker import drug_completeness_checker_ui
    drug_completeness_checker_ui()

elif page == "🔍 Drug Finder":
    from drug_finder import drug_finder_ui
    drug_finder_ui()

elif page == "🦠 AMR Checker":
    from amr_checker import amr_checker_ui
    amr_checker_ui()

elif page == "🧠 Pharm AI":
    from ollama_chat import ollama_chat_ui
    ollama_chat_ui()