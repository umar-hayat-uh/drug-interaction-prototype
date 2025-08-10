"""
Simple preprocessing script for drug interaction CSVs.

Usage:
    python src/data_preprocess.py --interactions data/db_drug_interactions.csv --drugbank data/drugbank_clean.csv --out data/processed.csv
"""
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def main(interactions_path, drugbank_path, out_path):
    # Read
    interactions = pd.read_csv(interactions_path, low_memory=False)
    drugbank = pd.read_csv(drugbank_path, low_memory=False)

    # Basic cleaning
    interactions = interactions.dropna(how='all')
    # Try to detect drug columns
    possible_a = [c for c in interactions.columns if 'drug' in c.lower()][:2]
    if len(possible_a) >= 2:
        a_col, b_col = possible_a[0], possible_a[1]
    else:
        # fallback column names
        a_col, b_col = 'drug1', 'drug2'

    interactions = interactions.rename(columns={a_col: 'drug1', b_col: 'drug2'})
    interactions = interactions[['drug1', 'drug2'] + [c for c in interactions.columns if c not in (a_col,b_col)]]

    # Keep a severity column if exists, otherwise create placeholder
    if 'severity' not in interactions.columns:
        interactions['severity'] = interactions.get('severity', 'unknown')

    # Encode drugs and severity
    le_drug = LabelEncoder()
    all_drugs = pd.Series(list(interactions['drug1'].dropna().unique()) + list(interactions['drug2'].dropna().unique()))
    le_drug.fit(all_drugs.astype(str))

    interactions['drug1_enc'] = le_drug.transform(interactions['drug1'].astype(str))
    interactions['drug2_enc'] = le_drug.transform(interactions['drug2'].astype(str))

    le_sev = LabelEncoder()
    interactions['severity_enc'] = le_sev.fit_transform(interactions['severity'].astype(str))

    # Optionally merge drug-level features from drugbank (e.g., average_mass)
    db_small = drugbank.loc[:, drugbank.columns.intersection(['drugbank-id','name','average-mass','monoisotopic-mass','targets','enzymes'])]
    db_small = db_small.rename(columns={'name':'drug_name'})
    # We'll keep this small; a richer merge requires normalization of names/ids
    # Save processed
    interactions.to_csv(out_path, index=False)
    print(f"Saved processed dataset to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactions', required=True)
    parser.add_argument('--drugbank', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.interactions, args.drugbank, args.out)
