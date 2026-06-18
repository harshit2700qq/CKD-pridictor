import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

print("Loading CKD Dataset...")

data = pd.read_csv("ckd.csv")


data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")


data.replace(["?", "nan", ""], np.nan, inplace=True)


for col in data.columns:
    try:
        data[col] = pd.to_numeric(data[col])
    except:
        pass


encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le


features = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv",
    "htn", "dm", "cad", "appet", "pe", "ane"
]

features = [f for f in features if f in data.columns]

X = data[features]
y = data["class"]


imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

model.fit(X, y)


os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(imputer, "models/imputer.pkl")
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(features, "models/features.pkl")

print("Model Ready ")