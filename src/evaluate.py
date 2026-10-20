"""Step 4: Evaluate models (confusion matrix, precision/recall/F1, ROC-AUC)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve)
from src import config


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    plt.figure(figsize=(7, 6))

    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })

        # Confusion matrix figure per model
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Diabetes", "Diabetes"],
                    yticklabels=["No Diabetes", "Diabetes"])
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR /
                    f"cm_{name.lower().replace(' ', '_')}.png", dpi=120)
        plt.close(fig)

        # Add to combined ROC plot
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.figure(1)
        plt.plot(fpr, tpr, label=f"{name} (AUC={rows[-1]['ROC-AUC']:.2f})")

    plt.figure(1)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "roc_curves.png", dpi=120)
    plt.close()

    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)\
        .round(3).reset_index(drop=True)
    results.to_csv(config.RESULTS_CSV, index=False)
    return results


def plot_feature_importance(model, feature_names):
    """Random Forest feature importance (Glucose & BMI usually on top)."""
    imp = pd.Series(model.feature_importances_, index=feature_names)\
        .sort_values()
    ax = imp.plot.barh(figsize=(6, 4), color="teal")
    ax.set_title("Random Forest - Feature Importance")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close()
