"""
Simple preprocessing script for drug interaction CSVs.

Usage:
    python src/data_preprocess.py --interactions data/db_drug_interactions.csv --drugbank data/drugbank_clean.csv --out data/processed.csv
Use: 
    python src/data_preprocess.py --interactions data/db_drug_interactions.csv --drugbank data/drugbank_clean.csv --out data/processed.csv
"""
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def infer_severity(desc):
    desc = str(desc).lower()
    if any(word in desc for word in ['severe', 'severely', 'risk', 'increase', 'adverse', 'high']):
        return 'high'
    elif any(word in desc for word in ['decrease', 'minor', 'medium', 'reduce', 'moderate', 'low']):
        return 'medium'
    else:
        return 'unknown'

def main(interactions_path, drugbank_path, out_path):
    interactions = pd.read_csv(interactions_path, low_memory=False)
    drugbank = pd.read_csv(drugbank_path, low_memory=False)

    interactions = interactions.dropna(how='all')

    possible_a = [c for c in interactions.columns if 'drug' in c.lower()][:2]
    if len(possible_a) >= 2:
        a_col, b_col = possible_a[0], possible_a[1]
    else:
        a_col, b_col = 'Drug 1', 'Drug 2'

    interactions = interactions.rename(columns={a_col: 'Drug 1', b_col: 'Drug 2'})
    interactions = interactions[['Drug 1', 'Drug 2'] + [c for c in interactions.columns if c not in (a_col,b_col)]]

    # Infer severity from 'Interaction Description' if severity column missing
    if 'severity' not in interactions.columns:
        if 'Interaction Description' in interactions.columns:
            interactions['severity'] = interactions['Interaction Description'].apply(infer_severity)
        else:
            interactions['severity'] = 'unknown'

    le_drug = LabelEncoder()
    all_drugs = pd.Series(list(interactions['Drug 1'].dropna().unique()) + list(interactions['Drug 2'].dropna().unique()))
    le_drug.fit(all_drugs.astype(str))

    interactions['Drug 1_enc'] = le_drug.transform(interactions['Drug 1'].astype(str))
    interactions['Drug 2_enc'] = le_drug.transform(interactions['Drug 2'].astype(str))

    le_sev = LabelEncoder()
    interactions['severity_enc'] = le_sev.fit_transform(interactions['severity'].astype(str))

    interactions.to_csv(out_path, index=False)
    print(f"Saved processed dataset to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactions', required=True)
    parser.add_argument('--drugbank', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.interactions, args.drugbank, args.out)
