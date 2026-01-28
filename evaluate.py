import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from models import MultiTaskNet
from data_processor import ClinicalPreprocessor

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model():
    print("🏥 Running Clinical Validation Protocol...")
    
    # 1. Load Data (Same logic as train.py)
    try:
        stroke_df = pd.read_csv('data/stroke.csv')
        lung_df = pd.read_csv('data/lung_cancer.csv')
    except FileNotFoundError:
        print("❌ Error: Data files not found.")
        return

    # Harmonize
    stroke_df.columns = [c.lower().replace(' ', '_') for c in stroke_df.columns]
    lung_df.columns = [c.lower().replace(' ', '_') for c in lung_df.columns]
    
    if 'level' in lung_df.columns:
        lung_df['lung_cancer'] = lung_df['level'].map({'High': 1, 'Medium': 1, 'Low': 0})
    if 'gender' in lung_df.columns and lung_df['gender'].dtype != object:
        lung_df['gender'] = lung_df['gender'].map({1: 'Male', 2: 'Female'})

    combined = pd.concat([stroke_df, lung_df], axis=0, ignore_index=True)
    combined['stroke'] = combined['stroke'].fillna(0)
    combined['lung_cancer'] = combined['lung_cancer'].fillna(0)

    # 2. Preprocess
    # We load the SAVED preprocessor to ensure we use the exact same scaling as the trained model
    try:
        pre = ClinicalPreprocessor.load('preprocessor.joblib')
        X = pre.transform(combined).values
        y_stroke = combined['stroke'].values
        y_lung = combined['lung_cancer'].values
    except Exception as e:
        print(f"❌ Error loading preprocessor: {e}")
        print("   (Run train.py first!)")
        return

    # 3. Load Trained Model
    try:
        model = torch.load('multitask_model.pth', map_location=DEVICE, weights_only=False)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 4. Evaluate
    print(f"📊 Evaluating on full dataset ({len(X)} records)...")
    
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        sp_logits, lp_logits, _ = model(X_tensor)
        
        # Predictions (Threshold 0.5)
        sp_pred = (torch.sigmoid(sp_logits) > 0.5).int().cpu().numpy()
        lp_pred = (torch.sigmoid(lp_logits) > 0.5).int().cpu().numpy()

    # 5. Print Report
    print("\n" + "="*40)
    print(f"🧠 STROKE MODEL ACCURACY: {accuracy_score(y_stroke, sp_pred):.2%}")
    print("="*40)
    print(classification_report(y_stroke, sp_pred, zero_division=0))

    print("\n" + "="*40)
    print(f"🫁 LUNG CANCER MODEL ACCURACY: {accuracy_score(y_lung, lp_pred):.2%}")
    print("="*40)
    print(classification_report(y_lung, lp_pred, zero_division=0))

if __name__ == "__main__":
    evaluate_model()