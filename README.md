# 🏦 Loan Approval Checker

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![License](https://img.shields.io/badge/license-MIT-green)

An interactive Streamlit app that estimates whether a loan application is **Likely Approved**, needs **Manual Review**, or is **Likely Not Approved**. It combines a Random Forest model with simple, transparent EMI and loan-to-income rules. The model is trained in the included Jupyter notebook.

> ⚠️ **For learning only.** This is not a real loan decision and guarantees nothing.

**Live demo:** _add your Streamlit Cloud link here_



## Features

- Instant result: move a slider or change a number and the decision updates.
- Colour-coded result card and a red / amber / green score meter.
- Estimated monthly EMI, rule score and ML probability shown side by side.
- "Why this result" breakdown of each scoring factor.
- Input checks that catch typos, such as entering lakhs instead of rupees.
- Amounts shown in lakh / crore for easy reading.

## How the decision is made

The final score (0–100) is a blend of the ML model and a rule score:

```
final = 25% × ML approval probability + 75% × rule score
```

**Rule score** (each factor is scaled from 0 to 100 points):

| Factor | Weight | 100 points at | 0 points at |
|--------|-------:|---------------|-------------|
| CIBIL score | 35% | 900 | 300 |
| EMI as share of monthly income | 40% | 30% | 70% |
| Loan as multiple of yearly income | 25% | 2× | 10× |

**Result bands**

| Final score | Result |
|-------------|--------|
| 70 or more | ✅ Likely Approved |
| 45 to 69 | 🟡 Manual Review |
| below 45 | ❌ Likely Not Approved |

**Safety rules that override the score**

- EMI above 60% of monthly income → Likely Not Approved
- Loan above 10× yearly income → Likely Not Approved
- CIBIL below 600 → result is capped at Manual Review (never rejected on CIBIL alone)

The EMI uses the standard formula with the interest rate and term you enter.

## The ML model

Trained in [`notebooks/loan_approval_model.ipynb`](notebooks/loan_approval_model.ipynb).

- **Data:** 4,269 loan applications, no missing values or duplicates, 62% approved.
- **Features (9):** dependents, annual income, loan amount, loan term, CIBIL score, and four asset values (residential, commercial, luxury, bank).
- **Dropped:** `loan_id` (identifier) and `education`, `self_employed` (chi-square p-values 0.77 and 1.0, so no link to approval).
- **Models compared:** Logistic Regression, Random Forest, Gradient Boosting, tuned with 5-fold `GridSearchCV` on ROC-AUC.
- **Selected:** Random Forest (best CV ROC-AUC, 0.9976).

| Model | Test accuracy | Test ROC-AUC |
|-------|--------------:|-------------:|
| Logistic Regression | 0.922 | 0.973 |
| **Random Forest** | **0.979** | **0.999** |
| Gradient Boosting | 0.984 | 0.998 |

Without `cibil_score`, the test ROC-AUC falls from 0.999 to 0.60. The model relies almost entirely on CIBIL, which is why it only carries 25% of the final score in the app. The notebook also includes permutation importance and SHAP plots.

## Limitations

- The dataset appears to be synthetic (near-perfect scores driven by one feature), so these metrics will not carry over to real lending.
- The rule weights and cut-offs are hand-picked and have not been validated on data.
- The dataset has no interest rate, so the EMI uses whatever rate you enter (default 10.5%).
- No fairness, bias or regulatory review has been done. Do not use this for real credit decisions.

## Project structure

```
loan-approval-checker/
├── app_interactive.py          # Streamlit app
├── models/loan_model.pkl       # trained Random Forest (scikit-learn 1.6.1)
├── notebooks/
│   └── loan_approval_model.ipynb
├── data/                       # put loan_approval_dataset.csv here (not committed)
├── docs/screenshot.png         # app screenshot used in this README
├── requirements.txt            # app dependencies
├── requirements-notebook.txt   # notebook dependencies
├── .streamlit/config.toml
├── .gitignore
├── LICENSE
└── README.md
```

## Run locally

```bash
git clone https://github.com/YOUR-USERNAME/loan-approval-checker.git
cd loan-approval-checker

python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app_interactive.py
```

The app opens at http://localhost:8501.

## Re-train the model

1. Download the **Loan Approval Prediction Dataset** from Kaggle and save it as `data/loan_approval_dataset.csv`. Check the dataset's licence before reusing it.
2. Install and open the notebook:

   ```bash
   pip install -r requirements-notebook.txt
   jupyter notebook notebooks/loan_approval_model.ipynb
   ```

3. The last cells save the model to `models/loan_model.pkl`. Run the notebook from the project root, or move the file there afterwards.

The pickle only loads with the scikit-learn version it was trained with (1.6.1). If you see a version error, re-train the model or install the pinned version.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (it must include `models/loan_model.pkl`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose this repo, branch `main`, and main file `app_interactive.py`.
4. Click **Deploy**, then paste the link at the top of this README.

## Tech stack

Python · Streamlit · scikit-learn · pandas · joblib · SHAP · Matplotlib / Seaborn

## License

MIT. See [LICENSE](LICENSE).
