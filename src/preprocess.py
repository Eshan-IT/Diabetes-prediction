"""Step 2: Preprocess the data (zeros -> NaN, impute, scale, split)."""
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src import config


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # 1. Replace impossible zeros with NaN
    df[config.ZERO_AS_MISSING] = df[config.ZERO_AS_MISSING].replace(0, np.nan)
    print("Missing values after zero->NaN:\n", df.isna().sum()[lambda s: s > 0])

    X = df.drop(columns=config.TARGET)
    y = df[config.TARGET]

    # 2. Split FIRST (so test data never leaks into imputer/scaler)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)

    # 3. Impute with median (fit on train only)
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_test_i = imputer.transform(X_test)

    # 4. Scale (fit on train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_i)
    X_test_s = scaler.transform(X_test_i)

    # Save fitted objects so the web app can reuse them
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(imputer, config.MODELS_DIR / "imputer.joblib")
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")

    # Save processed data for inspection
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(X.columns)
    pd.DataFrame(X_train_s, columns=cols).assign(Outcome=y_train.values)\
        .to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    pd.DataFrame(X_test_s, columns=cols).assign(Outcome=y_test.values)\
        .to_csv(config.PROCESSED_DIR / "test.csv", index=False)

    return X_train_s, X_test_s, y_train.values, y_test.values, cols
