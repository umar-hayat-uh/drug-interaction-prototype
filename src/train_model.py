"""
Train XGBoost model on processed drug interaction data.

Usage:
    python src/train_model.py --input data/processed.csv --model-out models/xgb_model.joblib
"""

import argparse
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score

def main(input_path, model_out):
    print("Loading data...")
    df = pd.read_csv(input_path)

    print("Preparing features and labels...")
    X = df[['Drug 1_enc', 'Drug 2_enc']].fillna(-1)
    y = df['severity_enc']

    num_classes = len(y.unique())
    print(f"Number of classes: {num_classes}")

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Initializing XGBClassifier...")
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss'
        # removed use_label_encoder because deprecated
    )

    print("Training model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    print("Predicting probabilities...")
    y_pred_proba = model.predict_proba(X_test)
    y_pred = y_pred_proba.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")

    print(f"Saving model to {model_out} ...")
    joblib.dump(model, model_out)
    print("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to processed CSV input')
    parser.add_argument('--model-out', required=True, help='Path to save trained model')
    args = parser.parse_args()
    main(args.input, args.model_out)
