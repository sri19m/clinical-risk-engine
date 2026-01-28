import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

class ClinicalPreprocessor:
    def __init__(self):
        # Handle missing numeric values with median
        self.num_imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        # Handle missing categorical values
        self.cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        self.fitted = False
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []

    def fit(self, df: pd.DataFrame, num_cols: list, cat_cols: list):
        self.numeric_features = num_cols
        self.categorical_features = cat_cols

        # 1. Numeric Pipeline
        # Force numeric conversion (coercing errors to NaN)
        num_df = df[self.numeric_features].apply(pd.to_numeric, errors='coerce')
        self.num_imputer.fit(num_df)
        num_imp = self.num_imputer.transform(num_df)
        self.scaler.fit(num_imp)

        # 2. Categorical Pipeline
        # Convert all to string to handle mixed types (e.g. Gender 1 vs "Male")
        cat_df = df[self.categorical_features].astype(str).fillna('Unknown')
        self.cat_imputer.fit(cat_df)
        cat_imp = self.cat_imputer.transform(cat_df)
        self.ohe.fit(cat_imp)

        # Save feature names
        self.feature_names = self.numeric_features + list(self.ohe.get_feature_names_out(self.categorical_features))
        self.fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted first.")

        # Ensure columns exist, fill missing with NaN
        for col in self.numeric_features + self.categorical_features:
            if col not in df.columns:
                df[col] = np.nan

        # Transform Numeric
        num_df = df[self.numeric_features].apply(pd.to_numeric, errors='coerce')
        num_data = self.scaler.transform(self.num_imputer.transform(num_df))
        
        # Transform Categorical
        cat_df = df[self.categorical_features].astype(str)
        cat_data = self.ohe.transform(self.cat_imputer.transform(cat_df))
        
        return pd.DataFrame(np.hstack([num_data, cat_data]), columns=self.feature_names)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)