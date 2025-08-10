# Drug Interaction Model Prototype

This repository is a lightweight prototype scaffold for the AI-powered drug interaction checker
(Science Fair project). It was made as a starting point — **you should adapt and
validate everything before using any output in a real/clinical setting**.

## What I included
- A project scaffold with `src/` scripts for preprocessing, training and a simple Streamlit app.
- `requirements.txt` listing dependencies (some optional/advanced packages may need manual install).
- This README with usage steps and notes.
- NOTE: Your provided CSVs (`db_drug_interactions.csv` and `drugbank_clean.csv`) were detected in the
  environment and can be used as input data.

## Files
- `src/data_preprocess.py` - reads raw CSV(s), performs cleaning and creates `data/processed.csv`.
- `src/train_model.py` - example training script (XGBoost) that reads `data/processed.csv` and saves a model to `models/`.
- `src/app_streamlit.py` - minimal Streamlit app to query two drug names and show a predicted severity.
- `requirements.txt` - Python dependencies.
- `notebooks/` - place for exploration / EDA notebooks.
- `data/` - place your raw CSVs here (or point the scripts to the uploads).
- `models/` - trained model artifacts will be saved here.

## Quick start (Ubuntu Jammy)
1. Create a python venv and activate:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

2. Install requirements (RDKit is optional and may require OS packages — see below):
```bash
pip install -r requirements.txt
```

3. Put your CSVs in `data/`:
- `data/db_drug_interactions.csv`  (or change path in scripts)
- `data/drugbank_clean.csv`

4. Preprocess:
```bash
python src/data_preprocess.py --interactions data/db_drug_interactions.csv --drugbank data/drugbank_clean.csv --out data/processed.csv
```

5. Train:
```bash
python src/train_model.py --input data/processed.csv --model-out models/xgb_model.joblib
```

6. Run the demo web app (Streamlit):
```bash
streamlit run src/app_streamlit.py
```

## Notes & caveats
- This is an educational prototype. **Not for clinical use.** Always validate with domain experts and gold-standard datasets.
- RDKit and some cheminformatics tools may require extra OS packages on Ubuntu. They are optional but useful for chemical fingerprint features.
- Some CSVs may be large — tune `read_csv` and memory usage accordingly.

## Next steps (suggested)
- Feature engineering: chemical fingerprints (RDKit), enzyme/target overlaps, ATC class similarity.
- Model improvements: hyperparameter tuning, calibration, multi-label handling.
- Validation: cross-validation, external test sets, precision/recall by severity.
- Explainability: SHAP to show which features drove predictions.
