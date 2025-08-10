"""
Example training script using XGBoost.

Usage:
    python src/train_model.py --input data/processed.csv --model-out models/xgb_model.joblib
"""
import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

def main(input_path, model_out):
    df = pd.read_csv(input_path)
    # Expect encoded columns created by the preprocess script
    X = df[['drug1_enc','drug2_enc']].fillna(-1)
    y = df['severity_enc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.4f}")
    joblib.dump(model, model_out)
    print(f"Saved model to {model_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--model-out', required=True)
    args = parser.parse_args()
    main(args.input, args.model_out)
