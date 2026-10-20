"""Step 3: Train multiple classification models."""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src import config


def get_models() -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", probability=True,
                   random_state=config.RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=config.RANDOM_STATE),
    }
    try:  # XGBoost is optional
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            eval_metric="logloss", random_state=config.RANDOM_STATE)
    except ImportError:
        print("xgboost not installed -> skipping XGBoost")
    return models


def train_all(X_train, y_train) -> dict:
    models = get_models()
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    return models
