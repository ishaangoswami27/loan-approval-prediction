import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Checker", page_icon="🏦")

MODEL_FILES = ["models/loan_model.pkl", "loan_model.pkl"]   # notebook saves to models/loan_model.pkl


@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    path = next((os.path.join(base, f) for f in MODEL_FILES if os.path.exists(os.path.join(base, f))), None)
    if path is None:
        raise FileNotFoundError(f"Place one of {MODEL_FILES} next to this file.")
    return joblib.load(path)


# ==== RULES START ===========================================================
# Score out of 100 from three parts. CIBIL is only 35% of it.
W_CIBIL, W_EMI, W_LOAN = 0.35, 0.40, 0.25
ML_WEIGHT = 0.25            # share of the ML model in the final score

# Cut-offs on the final score
APPROVE_AT, REVIEW_AT = 70, 45

# Safety rules
MAX_EMI_SHARE = 0.60        # EMI above 60% of monthly income -> Not Approved
MAX_LOAN_X_INCOME = 10      # loan above 10x yearly income    -> Not Approved
WEAK_CIBIL = 600            # below this: at best Manual Review (never rejected on CIBIL alone)


def scale(x, best, worst):
    """100 points at `best`, 0 points at `worst`, straight line in between."""
    return max(0.0, min(1.0, (x - worst) / (best - worst))) * 100


def monthly_emi(loan, rate_pct, years):
    n, r = int(years * 12), rate_pct / 1200
    return loan / n if r == 0 else loan * r * (1 + r) ** n / ((1 + r) ** n - 1)


def money(x):
    if x >= 1e7:
        return f"₹{x / 1e7:,.2f} crore"
    if x >= 1e5:
        return f"₹{x / 1e5:,.2f} lakh"
    return f"₹{x:,.0f}"


def rule_check(cibil, income, loan, term, rate):
    emi = monthly_emi(loan, rate, term)
    emi_share = emi / (income / 12)          # EMI as a share of monthly income
    loan_x = loan / income                   # loan as a multiple of yearly income
    parts = {
        "CIBIL score": (scale(cibil, 900, 300), W_CIBIL, f"{cibil}"),
        "EMI vs income": (scale(emi_share, 0.30, 0.70), W_EMI, f"{emi_share:.0%} of monthly income"),
        "Loan vs income": (scale(loan_x, 2, 10), W_LOAN, f"{loan_x:.1f}× yearly income"),
    }
    score = sum(pts * w for pts, w, _ in parts.values())
    return score, emi, emi_share, loan_x, parts


def decide(final, cibil, emi_share, loan_x):
    notes = []
    cat = "Likely Approved" if final >= APPROVE_AT else "Manual Review" if final >= REVIEW_AT else "Likely Not Approved"
    if emi_share > MAX_EMI_SHARE:
        cat = "Likely Not Approved"
        notes.append(f"The EMI would take {emi_share:.0%} of monthly income (limit {MAX_EMI_SHARE:.0%}).")
    if loan_x > MAX_LOAN_X_INCOME:
        cat = "Likely Not Approved"
        notes.append(f"The loan is {loan_x:.1f}× yearly income (limit {MAX_LOAN_X_INCOME}×).")
    if cat == "Likely Approved" and cibil < WEAK_CIBIL:
        cat = "Manual Review"
        notes.append(f"CIBIL is below {WEAK_CIBIL}, so the result is capped at Manual Review.")
    return cat, notes


def input_errors(income, loan):
    errs = []
    if income < 100_000:
        errs.append(f"Annual income {money(income)} is too low. Type the full amount in rupees "
                    "(e.g. 5,00,000 for ₹5 lakh), not in lakhs.")
    if loan < 100_000:
        errs.append(f"Loan amount {money(loan)} is too low. Type the full amount in rupees.")
    if income >= 100_000 and loan > 20 * income:
        errs.append(f"The loan is more than 20× the yearly income - that is not realistic. Please check.")
    return errs
# ==== RULES END =============================================================



# ---------------------------------------------------------------- look & feel
COLORS = {
    "Likely Approved": ("#30a46c", "✅"),
    "Manual Review": ("#f5a524", "🟡"),
    "Likely Not Approved": ("#e5484d", "❌"),
}


def result_card(category, final):
    color, icon = COLORS[category]
    return (
        f'<div style="padding:18px 22px;border-radius:14px;border-left:8px solid {color};'
        f'background:{color}22;margin:6px 0 14px 0">'
        f'<div style="font-size:1.7rem;font-weight:700">{icon} {category}</div>'
        f'<div style="opacity:.85;margin-top:2px">Final score <b>{final:.0f}</b> / 100</div></div>'
    )


def score_meter(final):
    """Red / amber / green bar with a pointer at the final score."""
    pos = max(0, min(100, final))
    red, amber, green = COLORS["Likely Not Approved"][0], COLORS["Manual Review"][0], COLORS["Likely Approved"][0]
    return (
        '<div style="margin:4px 0 2px 0">'
        f'<div style="position:relative;height:16px;border-radius:8px;background:linear-gradient(90deg,'
        f'{red} 0%,{red} {REVIEW_AT}%,{amber} {REVIEW_AT}%,{amber} {APPROVE_AT}%,{green} {APPROVE_AT}%,{green} 100%)">'
        f'<div style="position:absolute;left:{pos}%;top:-6px;width:6px;height:28px;border-radius:3px;'
        f'background:#fff;border:2px solid #222;transform:translateX(-50%);transition:left .5s"></div></div>'
        '<div style="display:flex;font-size:.78rem;opacity:.75;margin-top:6px">'
        f'<div style="width:{REVIEW_AT}%">Not approved</div>'
        f'<div style="width:{APPROVE_AT - REVIEW_AT}%">Review</div>'
        f'<div style="width:{100 - APPROVE_AT}%;text-align:right">Approved</div></div></div>'
    )


# ------------------------------------------------------------------- the page
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.info("Tip: `pip install scikit-learn==1.6.1`")
    st.stop()

st.title("🏦 Loan Checker")
st.caption("Move the sliders and the result updates instantly. It combines a Random Forest trained on 9 "
           "features (CIBIL, income, loan, term, dependents, 4 asset values) with simple EMI and loan-to-income rules.")

c1, c2 = st.columns(2)
with c1:
    cibil = st.slider("CIBIL score", 300, 900, 700)
    income = st.number_input("Annual income (₹)", min_value=0, value=5_000_000, step=100_000)
    st.caption(f"= {money(income)}")
with c2:
    loan = st.number_input("Loan amount (₹)", min_value=0, value=10_000_000, step=100_000)
    st.caption(f"= {money(loan)}")
    term = st.number_input("Loan term (years)", min_value=2, max_value=20, value=10)

with st.expander("More details (assets and dependents go to the ML model, interest rate is only for the EMI estimate)"):
    rate = st.number_input("Interest rate (% per year)", min_value=0.0, max_value=40.0, value=10.5, step=0.5)
    dependents = st.number_input("Dependents", min_value=0, max_value=5, value=2)
    residential = st.number_input("Residential assets (₹)", min_value=0, value=7_000_000, step=100_000)
    commercial = st.number_input("Commercial assets (₹)", min_value=0, value=4_000_000, step=100_000)
    luxury = st.number_input("Luxury assets (₹)", min_value=0, value=15_000_000, step=100_000)
    bank = st.number_input("Bank assets (₹)", min_value=0, value=5_000_000, step=100_000)

errors = input_errors(income, loan)
if errors:
    for e in errors:
        st.error(f"🚫 {e}")
    st.stop()

# ---- calculate (runs automatically on every change) ----
score, emi, emi_share, loan_x, parts = rule_check(cibil, income, loan, term, rate)
row = pd.DataFrame([{
    "no_of_dependents": dependents,
    "income_annum": income, "loan_amount": loan, "loan_term": term, "cibil_score": cibil,
    "residential_assets_value": residential, "commercial_assets_value": commercial,
    "luxury_assets_value": luxury, "bank_asset_value": bank,
}])
# education and self_employed were dropped in the notebook, so the model only uses these 9 columns
if hasattr(model, "feature_names_in_"):
    row = row[list(model.feature_names_in_)]
p_ml = float(model.predict_proba(row)[0][list(model.classes_).index(1)])  # class 1 = Approved
final = ML_WEIGHT * p_ml * 100 + (1 - ML_WEIGHT) * score
category, notes = decide(final, cibil, emi_share, loan_x)

# small pop-up when the result changes
previous = st.session_state.get("last_category")
if previous is not None and previous != category:
    st.toast(f"Result changed: {previous} → {category}", icon=COLORS[category][1])
st.session_state["last_category"] = category

# ---- show ----
st.divider()
st.markdown(result_card(category, final), unsafe_allow_html=True)
st.markdown(score_meter(final), unsafe_allow_html=True)
for n in notes:
    st.write(f"🛑 {n}")

st.write("")
a, b, c = st.columns(3)
a.metric("Estimated EMI", money(emi))
b.metric("Rule score", f"{score:.0f}/100")
c.metric("ML model", f"{p_ml:.0%}")

st.write("**Why this result**")
for name, (pts, weight, shown) in parts.items():
    st.write(f"{name} · {shown} · weight {weight:.0%}")
    st.progress(min(max(pts / 100, 0.0), 1.0), text=f"{pts:.0f}/100")

if p_ml * 100 < 10 or p_ml * 100 > 90:
    st.caption("Note: in the notebook the model's ROC-AUC fell from 0.999 to 0.60 without CIBIL, so it "
               "reacts mostly to CIBIL and counts for only 25% of the score here.")

st.caption("⚠️ For learning only. This is not a real loan decision and guarantees nothing.")
