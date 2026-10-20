<<<<<<< HEAD
# Diabetes Prediction (Supervised Binary Classification)

Predicts whether a patient is diabetic (`Outcome`: 0 = No, 1 = Yes) from the
Pima Indians Diabetes dataset (768 records).

## Folder structure
```
diabetes-prediction/
├── data/
│   ├── raw/          # original dataset (pima_raw.csv)
│   └── processed/    # cleaned + scaled train.csv / test.csv
├── src/              # one file per step
│   ├── config.py       # paths & settings
│   ├── data_loader.py  # Step 1: get dataset
│   ├── preprocess.py   # Step 2: zeros->NaN, impute, scale, split
│   ├── train.py        # Step 3: LR, KNN, SVM, Tree, RF, XGBoost
│   └── evaluate.py     # Step 4: confusion matrix, P/R/F1, ROC-AUC
├── models/           # saved best_model, imputer, scaler (.joblib)
├── reports/
│   ├── figures/        # confusion matrices, ROC curves, feature importance
│   └── model_comparison.csv
├── app/app.py        # Step 5: Streamlit web interface
├── main.py           # runs Steps 1-4 and saves the best model
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python main.py               # trains, evaluates, saves model
streamlit run app/app.py     # opens the prediction web app
```

## Notes
- Impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI
  are treated as missing and filled with the median.
- Imputer and scaler are fitted on the training split only (no data leakage)
  and saved so the web app applies identical preprocessing.
- Best model is chosen by ROC-AUC. Recall matters most for a screening task.
- Dataset source: Pima Indians Diabetes (UCI / Kaggle). 
=======
# Diabetes-prediction
Diabetes prediction web app using machine learning. Compares 5 classifiers on the Pima Indians dataset and deploys the best one (Random Forest, ROC-AUC 0.82) with Streamlit.

A supervised binary classification project that predicts whether a patient is diabetic from 8 health measurements (Glucose, BMI, Age, Insulin, etc.). It cleans the data, trains Logistic Regression, KNN, SVM, Decision Tree and Random Forest (plus XGBoost if installed), evaluates them with a confusion matrix, precision, recall, F1 and ROC-AUC, and serves the best model through a Streamlit app.

⚠️ Educational project, not a medical diagnosis.

