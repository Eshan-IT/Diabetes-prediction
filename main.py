"""Run the full pipeline:  python main.py"""
import joblib
from src import config
from src.data_loader import load_raw_data
from src.preprocess import preprocess
from src.train import train_all
from src.evaluate import evaluate_all, plot_feature_importance


def main():
    df = load_raw_data()
    X_train, X_test, y_train, y_test, features = preprocess(df)
    models = train_all(X_train, y_train)

    results = evaluate_all(models, X_test, y_test)
    print("\n=== Model Comparison (sorted by ROC-AUC) ===")
    print(results.to_string(index=False))

    plot_feature_importance(models["Random Forest"], features)

    # Pick best model by ROC-AUC (best at separating the two classes)
    best_name = results.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    joblib.dump(models[best_name], config.MODELS_DIR / "best_model.joblib")
    joblib.dump(features, config.MODELS_DIR / "feature_names.joblib")
    print(f"\nBest model saved: {best_name} -> models/best_model.joblib")


if __name__ == "__main__":
    main()
