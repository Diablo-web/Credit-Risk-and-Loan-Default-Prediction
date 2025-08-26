# pages/1_Loan_Predictor.py
import os, json, base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_shap import st_shap
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# ---------- Page + global style ----------
st.set_page_config(page_title="NextGenCredit — Credit Scoring", layout="wide")

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE if HERE.name != "pages" else HERE.parent

# Load your global CSS (Manrope + UI polish)
css_file = APP_ROOT / "assets" / "css" / "styles.css"
if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# --- Floating top-right GIF ---
from pathlib import Path
from base64 import b64encode
import streamlit as st

GIF_PATH = Path("assets/img/coins.gif")   # change if you use another name/path

if GIF_PATH.exists():
    b64 = b64encode(GIF_PATH.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
          /* Make the main content container a positioning context */
          [data-testid="stAppViewContainer"] .main .block-container {{
            position: relative;
          }}
          /* Place the GIF at the top-right INSIDE the content area */
          .corner-gif {{
            position: absolute;
            top:12px;                 /* adjust vertical offset */
            right: 12px;               /* adjust horizontal offset */
            z-index: 15;               /* above content, below toolbar */
            pointer-events: none;      /* don't block clicks */
          }}
          .corner-gif img {{
           
            width: clamp(120px, 16vw, 220px);
            height: auto;
            display: block;
            border-radius: 8px;        /* optional */
            filter: drop-shadow(0 4px 10px rgba(0,0,0,.15));
          }}
          /* Hide on very small screens to avoid crowding */
          
          @media (max-width: 700px) {{
          .corner-gif img{{ width: clamp(90px, 24vw, 140px); }}
            
          }}
        </style>
        <div class="corner-gif">
          <img src="data:image/gif;base64,{b64}" alt="coins" />
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.caption(" Place your GIF at `assets/img/coins.gif` to show it in the corner.")


# ---------- Model + artifacts ----------
def _load(p):
    try:
        import joblib as _joblib
        return _joblib.load(p)
    except Exception:
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_pipeline(path="models/xgb_pipeline.pkl"):
    return _load(path)

@st.cache_resource(show_spinner=False)
def load_background(path="models/bg_sample.parquet"):
    return pd.read_parquet(path) if os.path.exists(path) else None

@st.cache_resource(show_spinner=False)
def load_threshold(path="models/threshold.json", default=0.50):
    try:
        with open(path) as f: 
            return float(json.load(f).get("threshold", default))
    except Exception:
        return float(default)

pipe = load_pipeline()
bg   = load_background()
thr_default = load_threshold()

# ---------- Schema (must match training) ----------
NUMERIC = [
    "loan_amnt","term_months","int_rate","installment","annual_inc","dti",
    "open_acc","pub_rec","inq_last_6mths","revol_bal","revol_util","total_acc","emp_length_yrs"
]
CATEG  = ["home_ownership","verification_status","purpose","application_type"]
ALL    = NUMERIC + CATEG

# ---------- Header ----------
st.title("NextGenCredit")
st.caption("Analyst view with bank-style inputs and explainability")

# ---------- Inputs ----------
st.subheader("Borrower & Loan Details")
c1, c2 = st.columns(2)
with c1:
    loan_amnt   = st.number_input("Loan amount", 500, 200_000, 6_000, step=100)
    term_months = st.selectbox("Term (months)", [36, 60], index=0)
    int_rate    = st.number_input("Interest rate (%)", 0.0, 60.0, 18.0, step=0.1)
    installment = st.number_input("Monthly installment", 1.0, 5_000.0, 220.0, step=1.0)
    annual_inc  = st.number_input("Annual income", 0.0, 5_000_000.0, 120_000.0, step=1000.0)
    dti         = st.number_input("DTI", 0.0, 200.0, 25.0, step=0.1)
with c2:
    open_acc       = st.number_input("Open credit lines", 0, 200, 12)
    pub_rec        = st.number_input("Public derogatories", 0, 50, 0)
    inq_last_6mths = st.number_input("Inquiries (6 months)", 0, 50, 1)
    revol_bal      = st.number_input("Revolving balance", 0.0, 5_000_000.0, 8_000.0, step=100.0)
    revol_util     = st.number_input("Revolving util (%)", 0.0, 300.0, 35.0, step=0.1)
    total_acc      = st.number_input("Total credit lines", 0, 300, 30)
emp_label = st.selectbox(
        "Employment length (years)",
        ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10+"],
        index=5  # default value = "5"
    )

    # Map to numeric for model
emp_length_yrs = 10 if emp_label == "10+" else int(emp_label)
# removed slicer emp_length_yrs = st.slider("Employment length (years)", 0, 10, 5)

c3, c4 = st.columns(2)
with c3:
    home_ownership = st.selectbox("Home ownership", ["MORTGAGE","RENT","OWN","OTHER"])
    verification_status = st.selectbox("Verification status", ["Verified","Source Verified","Not Verified"])
with c4:
    application_type = st.selectbox("Application type", ["INDIVIDUAL","JOINT"])
    purpose = st.selectbox(
        "Purpose",
        ["debt_consolidation","credit_card","home_improvement","major_purchase","small_business","car",
         "medical","wedding","house","moving","vacation","educational","other"]
    )

X_user = pd.DataFrame([{
    "loan_amnt": loan_amnt, "term_months": term_months, "int_rate": int_rate,
    "installment": installment, "annual_inc": annual_inc, "dti": dti,
    "open_acc": open_acc, "pub_rec": pub_rec, "inq_last_6mths": inq_last_6mths,
    "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
    "emp_length_yrs": emp_length_yrs, "home_ownership": home_ownership,
    "verification_status": verification_status, "purpose": purpose, "application_type": application_type
}])[ALL]

# ---------- Tabs ----------
tab_pred, tab_explain = st.tabs([" Check", "Explain (SHAP & LIME)"])



# ======== Predict tab (minimal) ========
with tab_pred:
    # compact button (no threshold popover)


    btn_col, _ = st.columns([0.28, 0.72])
    go = btn_col.button("Predict", type="primary", use_container_width=True)

    if go:
        # (optional) quick policy screen first
        
        den = max(float(annual_inc), 1e-6)
        reasons = []
        if (loan_amnt / den) >= 0.70:
            reasons.append("Loan amount ≥ 70% of annual income")
        if (revol_bal / den) >= 0.30:
            reasons.append("Revolving balance ≥ 30% of annual income")

        if reasons:
            st.error(" It cannot pay the loan — policy limits.")
            st.caption(" · ".join(reasons))
        else:
            # model prediction (PD kept internal)
            pd_default = float(pipe.predict_proba(X_user)[:, 1][0])
            THRESHOLD = 0.50  # fixed, hidden

            # final verdict (binary)
            if pd_default >= THRESHOLD:
                st.error("Predicted: Default risk high — Decline", icon="⚠️")
            else:
                st.success("Predicted: Default risk acceptable — Approve")

            # show only estimated credit score (300–850 style)
            score_850 = int(np.clip(round(850 - pd_default * 550), 300, 850))
            st.metric("Estimated credit score", f"{score_850}")

        # inputs only (no PD/threshold shown)
        with st.expander("Show inputs"):
            st.write(X_user)


# ======== Explain tab (SHAP fast path) ========


# ---------- Explain ----------# 
with tab_explain:
    st.caption("Global + local explanations. SHAP shows feature pushes; LIME shows top local rules.")
    if bg is None or bg.empty:
        st.warning(
            "No background sample found at models/bg_sample.parquet. "
            "SHAP/LIME will be slow without it. Save a ~200–1000 row sample during training."
        )
    else:
        # 1) Pipeline parts
        pre = pipe.named_steps.get("preprocess") or pipe.named_steps.get("prep")
        est = pipe.named_steps.get("model")      or pipe.named_steps.get("clf")
        if pre is None or est is None:
            st.error("Pipeline must contain a preprocessing step ('preprocess'/'prep') and a model ('model'/'clf').")
            st.stop()

        # 2) RAW -> model space
        Xbg_tr = pre.transform(bg[ALL])
        X1_tr  = pre.transform(X_user[ALL])

        # Dense float32 for plotting
        if hasattr(Xbg_tr, "toarray"): Xbg_tr = Xbg_tr.toarray()
        if hasattr(X1_tr,  "toarray"): X1_tr  = X1_tr.toarray()
        Xbg_tr = np.asarray(Xbg_tr, dtype=np.float32)
        X1_tr  = np.asarray(X1_tr,  dtype=np.float32)

        # Feature names after preprocessing (incl. OHE)
        try:
            feat_names = pre.get_feature_names_out(ALL)
        except Exception:
            try:
                feat_names = pre.get_feature_names_out()
            except Exception:
                feat_names = np.array([f"f{i}" for i in range(Xbg_tr.shape[1])])

        # 3) SHAP (TreeExplainer on estimator, FAST)
        explainer = shap.TreeExplainer(
            est,
            data=Xbg_tr,  # background keeps interventional path
            feature_perturbation="interventional",
            model_output="probability",
        )

        # Subsample background for the global plot (speed)
        n = Xbg_tr.shape[0]
        k = min(2000, n)
        if n > k:
            idx = np.random.default_rng(42).choice(n, size=k, replace=False)
            Xbg_plot = Xbg_tr[idx]
        else:
            Xbg_plot = Xbg_tr

        # Compute SHAP values once
        sv_bg = explainer.shap_values(Xbg_plot, check_additivity=False)
        sv_bg = sv_bg[1] if isinstance(sv_bg, list) else sv_bg  # class-1

        # --- SHAP Global (beeswarm) ---
        st.subheader("SHAP — Global importance")
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        plt.sca(ax1)  # set current axes for SHAP
        shap.summary_plot(sv_bg, features=Xbg_plot, feature_names=feat_names, show=False)
        st.pyplot(fig1, clear_figure=True)

        # --- SHAP Local (waterfall for current input) ---
        st.subheader("SHAP — Local (waterfall)")
        sv_one = explainer.shap_values(X1_tr, check_additivity=False)
        sv1    = sv_one[1] if isinstance(sv_one, list) else sv_one
        base   = explainer.expected_value[1] if isinstance(explainer.expected_value,(list,np.ndarray)) else explainer.expected_value
        ex     = shap.Explanation(values=sv1[0], base_values=base, data=X1_tr[0], feature_names=feat_names)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        plt.sca(ax2)
        shap.plots.waterfall(ex, max_display=20, show=False)
        st.pyplot(fig2, clear_figure=True)

        # 4) LIME (local)
        # ---- LIME (local) — use transformed numeric matrix to avoid strings ----
        st.subheader("LIME — Local (top 10)")
        from lime.lime_tabular import LimeTabularExplainer

        lime_explainer = LimeTabularExplainer(
            training_data=Xbg_tr,                 # numeric, after preprocessing
            feature_names=list(feat_names),       # expanded names (incl. one-hot)
            class_names=["Fully Paid","Default"],
            mode='classification',
            discretize_continuous=True
        )

        # predict directly on the estimator with transformed inputs
        lime_exp = lime_explainer.explain_instance(
            data_row=X1_tr[0],
            predict_fn=lambda Z: est.predict_proba(Z),
            num_features=10
        )
        st.pyplot(lime_exp.as_pyplot_figure(), clear_figure=True)

        st.info(
            "Reading the charts:\n"
            "• SHAP beeswarm ranks features by global impact; right = higher default risk, left = lower.\n"
            "• SHAP waterfall explains this borrower: red bars raise risk, blue bars lower it.\n"
            "• LIME shows top local rules that support the decision."
        )



