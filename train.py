import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib
from models import Autoencoder, MultiTaskNet
from data_processor import ClinicalPreprocessor

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def load_and_clean_data():
    print("[1/5] Loading & Harmonizing Data...")
    try:
        stroke_df = pd.read_csv('data/stroke.csv')
        lung_df = pd.read_csv('data/lung_cancer.csv')
    except FileNotFoundError:
        print("❌ Error: Files not found!")
        return None

    # Standardize columns
    stroke_df.columns = [c.lower().replace(' ', '_') for c in stroke_df.columns]
    lung_df.columns = [c.lower().replace(' ', '_') for c in lung_df.columns]

    # Map Lung Cancer Target
    if 'level' in lung_df.columns:
        lung_df['lung_cancer'] = lung_df['level'].map({'High': 1, 'Medium': 1, 'Low': 0})
    
    # Map Gender
    if 'gender' in lung_df.columns:
        lung_df['gender'] = lung_df['gender'].map({1: 'Male', 2: 'Female'})

    # Combine
    combined = pd.concat([stroke_df, lung_df], axis=0, ignore_index=True)
    
    # Fill Target NaNs
    combined['stroke'] = combined['stroke'].fillna(0)
    combined['lung_cancer'] = combined['lung_cancer'].fillna(0)
    
    return combined

def train_pipeline():
    combined_raw = load_and_clean_data()
    if combined_raw is None: return

    # --- THE MASSIVE UPGRADE ---
    # We are now using 24 features instead of 8!
    numeric_cols = [
        # Vitals
        'age', 'bmi', 'avg_glucose_level',
        # Environment/Habits
        'air_pollution', 'alcohol_use', 'genetic_risk', 'obesity', 'smoking', 'passive_smoker',
        # Symptoms (Lung)
        'chest_pain', 'coughing_of_blood', 'fatigue', 'weight_loss', 'shortness_of_breath',
        'wheezing', 'swallowing_difficulty', 'clubbing_of_finger_nails', 'frequent_cold',
        'dry_cough', 'snoring', 
        # History (Stroke)
        'hypertension', 'heart_disease'
    ]
    
    # Categorical
    cat_cols = ['gender', 'smoking_status', 'work_type', 'residence_type']

    print(f"[2/5] Training on {len(numeric_cols) + len(cat_cols)} distinct features...")

    # Preprocess
    pre = ClinicalPreprocessor()
    # Filter to ensure cols actually exist
    valid_num = [c for c in numeric_cols if c in combined_raw.columns]
    valid_cat = [c for c in cat_cols if c in combined_raw.columns]
    
    pre.fit(combined_raw, valid_num, valid_cat)
    X = pre.transform(combined_raw).values
    pre.save('preprocessor.joblib')
    
    # Train Autoencoder
    print(f"[3/5] Training Autoencoder ({X.shape[1]} inputs)...")
    ae = Autoencoder(input_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    tensor_x = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    
    for epoch in range(50):
        ae.train()
        optimizer.zero_grad()
        recon, _ = ae(tensor_x)
        loss = loss_fn(recon, tensor_x)
        loss.backward()
        optimizer.step()
    torch.save(ae, 'autoencoder.pth')

    # Clustering
    ae.eval()
    with torch.no_grad(): _, latents = ae(tensor_x)
    kmeans = KMeans(n_clusters=3, random_state=SEED).fit(latents.cpu().numpy())
    joblib.dump(kmeans, 'kmeans.joblib')

    # Train Multi-Task Net
    print("[4/5] Training Multi-Task Network...")
    y_stroke = combined_raw['stroke'].values
    y_lung = combined_raw['lung_cancer'].values
    
    mt_net = MultiTaskNet(input_dim=X.shape[1]).to(DEVICE)
    optimizer_mt = torch.optim.Adam(mt_net.parameters(), lr=1e-3)
    
    y_s_t = torch.tensor(y_stroke, dtype=torch.float32).to(DEVICE)
    y_l_t = torch.tensor(y_lung, dtype=torch.float32).to(DEVICE)
    y_c_t = torch.tensor(kmeans.labels_, dtype=torch.long).to(DEVICE)
    
    # Add Class Weight to fix Stroke Imbalance (20x weight for strokes)
    pos_weight = torch.tensor([20.0]).to(DEVICE)
    
    for epoch in range(60):
        mt_net.train()
        optimizer_mt.zero_grad()
        s_pred, l_pred, c_pred = mt_net(tensor_x)
        
        loss_s = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(s_pred, y_s_t) # Weighted!
        loss_l = nn.BCEWithLogitsLoss()(l_pred, y_l_t)
        loss_c = nn.CrossEntropyLoss()(c_pred, y_c_t)
        
        (loss_s + loss_l + 0.1 * loss_c).backward()
        optimizer_mt.step()

    torch.save(mt_net, 'multitask_model.pth')
    print("✅ UPGRADE COMPLETE: System is now tracking 25+ medical variables.")

if __name__ == "__main__":
    train_pipeline()