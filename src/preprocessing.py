import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame, target_column: str):
    data = df.copy()
    X = pd.get_dummies(data.drop(columns=[target_column]), drop_first=False, dtype=float)
    y = data[target_column]
    if y.dtype == "object":
        y = pd.factorize(y)[0]
    y = pd.Series(y, index=data.index)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y), scaler, X.columns.tolist()
