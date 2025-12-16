
# Telco Customer Churn — Binary Classification

This project implements a binary classification pipeline to predict customer churn from the Telco dataset.

**Binary Classification**
- **Description:** Predict whether a customer will churn (Yes / No).
- **Target mapping:** `Churn` is encoded as 1 = Yes, 0 = No.
- **Dataset:** `Telco-Customer-Churn.csv` (features include demographics, services, charges, contract and payment info).

**Preprocessing summary**
- **Drop ID:** `customerID` is removed.
- **Numeric conversion:** `TotalCharges` converted to numeric and missing values filled (median).
- **Binary cols:** Encode binary categorical columns (e.g., `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) as 0/1.
- **Multi-category:** One-hot encode multi-category columns (e.g., `Contract`, `PaymentMethod`, `InternetService`, etc.).
- **Scaling:** Standardize numerical features (e.g., `tenure`, `MonthlyCharges`, `TotalCharges`).

**Model & Training**
- A Keras model is defined and tuned with `keras_tuner` (Hyperband) to search layer sizes and learning rate.
- The notebook trains the model, evaluates on a hold-out test set, and saves the final model to `churn_model.keras`.

**Evaluation metrics & considerations**
- **Metrics used:** Accuracy, confusion matrix, classification report (precision/recall/F1), ROC AUC.
- **Class imbalance:** The dataset contains more non-churn examples; consider class weighting, oversampling (SMOTE), or threshold tuning to improve recall for the minority class.
- **Dropout:** Not used because overfitting not occured. Train and Test accuracies not showed significant difference.

**How to run**
- Open and run `Binary_Classification.ipynb` in JupyterLab / Jupyter Notebook / Google Colab.
- Ensure Python packages in the notebook (TensorFlow, keras_tuner, scikit-learn, pandas, numpy, matplotlib, seaborn) are installed.

For implementation details and code, see `Binary_Classification.ipynb` in this repository.

