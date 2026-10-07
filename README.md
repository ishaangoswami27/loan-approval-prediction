# Loan Approval Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a loan application will be **Approved** or **Rejected** using machine learning classification algorithms.

The project performs data preprocessing, exploratory data analysis (EDA), statistical analysis, feature selection, model training, hyperparameter tuning, model evaluation, and model saving.

The trained model can also be used to make predictions for new loan applicants.

---

## 🎯 Project Objective

The main objective is to build a machine learning model that can predict loan approval based on applicant and loan-related information.

The target variable is:

* `loan_status`

  * `1` = Approved
  * `0` = Rejected

---

## 📂 Project Structure

```text
loan-approval-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── loan_approval_analysis.ipynb
│
├── data/
│   └── loan_approval_dataset.csv
│
├── models/
│   └── loan_model.pkl
│
└── app.py
```

---

## 📊 Dataset

The dataset contains information related to loan applicants and their applications.

Important features used in the project include:

* `no_of_dependents`
* `income_annum`
* `loan_amount`
* `loan_term`
* `cibil_score`
* `residential_assets_value`
* `commercial_assets_value`
* `luxury_assets_value`
* `bank_asset_value`

The following columns are handled separately:

* `loan_id` is removed because it is an identifier.
* `loan_status` is used as the target variable.
* `education` and `self_employed` are evaluated using statistical analysis and may be removed depending on the p-value.

---

## 🔧 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## 🔄 Machine Learning Workflow

The project follows these major steps:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Statistical Analysis
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Model Training
      ↓
GridSearchCV
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Model Saving
```

---

## 🧹 Data Preprocessing

The following preprocessing operations are performed:

1. Remove extra spaces from column names.
2. Remove extra spaces from categorical values.
3. Convert categorical values into numerical values.
4. Check missing values.
5. Check duplicate records.
6. Analyze class distribution.
7. Remove `loan_id`.
8. Evaluate categorical features using Chi-square testing.
9. Split the dataset into training and testing sets.

The dataset is divided using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

---

## 📈 Exploratory Data Analysis

The notebook performs several visual analyses, including:

* Loan approval distribution
* Histograms of numerical variables
* Correlation heatmap
* CIBIL score distribution by loan status
* Loan amount distribution by loan status
* Approval rate by education
* Approval rate by self-employment status

---

## 🤖 Machine Learning Models

Three classification algorithms are evaluated:

### 1. Logistic Regression

Logistic Regression is implemented with:

* StandardScaler
* Class balancing
* GridSearchCV

### 2. Random Forest

Random Forest is tuned using parameters such as:

* `n_estimators`
* `max_depth`
* `min_samples_leaf`

### 3. Gradient Boosting

Gradient Boosting is tuned using:

* `n_estimators`
* `learning_rate`
* `max_depth`

---

## 🔍 Hyperparameter Tuning

`GridSearchCV` is used with 5-fold cross-validation.

The optimization metric is:

```text
ROC-AUC
```

The model with the highest cross-validation ROC-AUC is selected as the best model.

---

## 📊 Model Evaluation

The project evaluates the models using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix
* ROC Curve

The project also compares training and testing ROC-AUC to help identify possible overfitting.

---

## ⭐ Feature Importance

Permutation importance is used to understand which features contribute most to the performance of the selected model.

The project also performs an additional experiment by removing:

```text
cibil_score
```

and comparing the resulting ROC-AUC with the original model.

---

## 💾 Saved Model

The final selected model is saved using Joblib:

```python
joblib.dump(best_model, "models/loan_model.pkl")
```

The saved model can later be loaded with:

```python
import joblib

model = joblib.load("models/loan_model.pkl")
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-ml.git
```

### 2. Open the project

```bash
cd loan-approval-ml
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/loan_approval_analysis.ipynb
```

---

## 📌 Prediction Example

After loading the trained model:

```python
model = joblib.load("models/loan_model.pkl")

prediction = model.predict(new_data)
probability = model.predict_proba(new_data)
```

The prediction can be interpreted as:

```text
1 → Loan Approved
0 → Loan Rejected
```

---

## ⚠️ Disclaimer

This project is intended for **educational and portfolio purposes**.

A real-world loan approval system should consider additional factors, regulatory requirements, fairness, explainability, data quality, and appropriate human oversight before being used for actual lending decisions.

---

## 👨‍💻 Author

**Ishaan Goswami**

Data Science / Machine Learning Project

---

## ⭐ Future Improvements

Possible improvements include:

* Build a Streamlit web application.
* Add FastAPI model serving.
* Add automated testing.
* Add model versioning.
* Add Docker support.
* Add CI/CD using GitHub Actions.
* Improve feature engineering.
* Add model explainability using SHAP.
* Monitor model performance after deployment.

---

## 📜 License

This project can be used for educational and portfolio purposes.
