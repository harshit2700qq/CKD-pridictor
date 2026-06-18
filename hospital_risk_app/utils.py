import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_and_clean_data(path):
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.replace("'", "")
    df.replace(["?", "nan", ""], np.nan, inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    return df


def encode_data(df):
    encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders