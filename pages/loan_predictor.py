# # # file: pages/1_Loan_Predictor.py
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib

# # st.set_page_config(page_title="Loan Predictor", page_icon="🔎", layout="centered")

# # st.title("🔎 Loan Default Predictor")
# # st.caption("Enter borrower & loan details to estimate default risk. The app loads a saved scikit-learn Pipeline (preprocessing + model).")

# # # -------- Load trained pipeline (cached) --------
# # @st.cache_resource(show_spinner=False)
# # def load_pipeline():
# #     return joblib.load("models/logistic_regression_model_1.2.pkl")  # path relative to project root

# # try:
# #     pipe = load_pipeline()
# # except Exception as e:
# #     st.error("Could not load the saved pipeline at **models/loan_pipeline.joblib**.")
# #     st.exception(e)
# #     st.stop()

# # # -------- Feature schema (must match your training) --------
# # NUMERIC_FEATURES = [
# #     "loan_amnt","term_months","int_rate","installment","annual_inc","dti",
# #     "open_acc","pub_rec","inq_last_6mths","revol_bal","revol_util","total_acc","emp_length_yrs"
# # ]
# # CATEGORICAL_FEATURES = [
# #     "home_ownership","verification_status","purpose","application_type"
# # ]
# # ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# # # -------- Sidebar: Threshold --------
# # st.sidebar.header("Decision Threshold")
# # thr = st.sidebar.slider("Flag as default if probability ≥", 0.10, 0.70, 0.30, 0.01)
# # st.sidebar.caption("Tip: Lower threshold → catch more defaulters (recall↑) but more false alarms (precision↓).")

# # # -------- Form inputs --------
# # st.subheader("Borrower & Loan Details")

# # col1, col2 = st.columns(2)
# # with col1:
# #     loan_amnt     = st.number_input("Loan amount (USD)", min_value=500, max_value=100_000, value=6_000, step=100)
# #     term_months   = st.selectbox("Term (months)", [36, 60], index=0)
# #     int_rate      = st.number_input("Interest rate (%)", min_value=0.0, max_value=60.0, value=17.5, step=0.1)
# #     installment   = st.number_input("Monthly installment (USD)", min_value=1.0, max_value=5_000.0, value=220.0, step=1.0)
# #     annual_inc    = st.number_input("Annual income (USD)", min_value=0.0, max_value=2_000_000.0, value=150_000.0, step=1000.0)
# #     dti           = st.number_input("Debt-to-income (DTI)", min_value=0.0, max_value=200.0, value=24.0, step=0.1)

# # with col2:
# #     open_acc        = st.number_input("Open credit lines", min_value=0, max_value=200, value=15, step=1)
# #     pub_rec         = st.number_input("Public derogatories", min_value=0, max_value=50, value=0, step=1)
# #     inq_last_6mths  = st.number_input("Inquiries (last 6 months)", min_value=0, max_value=50, value=1, step=1)
# #     revol_bal       = st.number_input("Revolving balance (USD)", min_value=0.0, max_value=2_000_000.0, value=8_000.0, step=100.0)
# #     revol_util      = st.number_input("Revolving utilization (%)", min_value=0.0, max_value=300.0, value=35.0, step=0.1)
# #     total_acc       = st.number_input("Total credit lines", min_value=0, max_value=300, value=34, step=1)

# # emp_length_yrs = st.slider("Employment length (years)", 0, 10, 10, help="10 means 10+ years")

# # col3, col4 = st.columns(2)
# # with col3:
# #     home_ownership = st.selectbox("Home ownership", ["MORTGAGE","RENT","OWN","OTHER"])
# #     verification_status = st.selectbox("Verification status", ["Verified","Source Verified","Not Verified"])
# # with col4:
# #     application_type = st.selectbox("Application type", ["INDIVIDUAL","JOINT"])
# #     purpose = st.selectbox(
# #         "Loan purpose",
# #         ["debt_consolidation","credit_card","home_improvement","major_purchase","small_business","car",
# #          "medical","wedding","house","moving","vacation","educational","other"]
# #     )

# # # Build the single-row input to match training
# # X_user = pd.DataFrame([{
# #     "loan_amnt": loan_amnt,
# #     "term_months": term_months,
# #     "int_rate": int_rate,
# #     "installment": installment,
# #     "annual_inc": annual_inc,
# #     "dti": dti,
# #     "open_acc": open_acc,
# #     "pub_rec": pub_rec,
# #     "inq_last_6mths": inq_last_6mths,
# #     "revol_bal": revol_bal,
# #     "revol_util": revol_util,
# #     "total_acc": total_acc,
# #     "emp_length_yrs": emp_length_yrs,
# #     "home_ownership": home_ownership,
# #     "verification_status": verification_status,
# #     "purpose": purpose,
# #     "application_type": application_type
# # }])[ALL_FEATURES]  # enforce column order

# # st.markdown("---")
# # if st.button("Predict"):
# #     try:
# #         proba_default = float(pipe.predict_proba(X_user)[:, 1][0])
# #         pred_label = int(proba_default >= thr)

# #         st.metric("Predicted probability of default", f"{proba_default:.2%}")
# #         if pred_label == 1:
# #             st.error(f"⚠️ Predicted: Charged Off (Default)  •  threshold = {thr:.2f}")
# #         else:
# #             st.success(f"✅ Predicted: Fully Paid  •  threshold = {thr:.2f}")

# #         with st.expander("Show model inputs"):
# #             st.write(X_user)

# #         with st.expander("What does the threshold mean?"):
# #             st.write(
# #                 "If the model's default probability is **≥ threshold**, we flag as **Default (1)**.\n\n"
# #                 "Lower threshold → higher recall (catch more defaulters) but lower precision.\n"
# #                 "Higher threshold → higher precision but lower recall."
# #             )

# #     except Exception as e:
# #         st.error("Prediction failed. Make sure your saved pipeline expects these feature names and categories.")
# #         st.exception(e)


# import streamlit as st
# import pandas as pd, numpy as np, json, joblib
# from pathlib import Path
# import joblib



# st.set_page_config(page_title="Loan Default – RF", page_icon="", layout="centered")
# st.title("XGBoost – Loan Default Predictor")

# # import psutil

# # mem = psutil.virtual_memory()
# # print(f"Available memory: {mem.available / (1024 ** 2):.2f} MB")

# @st.cache_resource(show_spinner=False)
# def load_model_memmapped(path: str = "models/rf_pipeline.pkl"):
#     # Memory-map the numpy arrays inside the pickle to avoid big allocations
#     return joblib.load(path, mmap_mode="r")

# # --- Load model + tuned threshold + background for SHAP ---
# pipe = load_model_memmapped("models/rf_pipeline.pkl")

# # pipe = joblib.load("models/rf_pipeline.pkl")

# thr_default = 0.50
# m = Path("models/rf_metrics.json")
# if m.exists():
#     with open(m) as f:
#         thr_default = json.load(f)["threshold"]

# bg = None
# p = Path("models/background.parquet")
# if p.exists():
#     bg = pd.read_parquet(p)

# # --- Schema (must match what you trained on) ---
# NUMERIC = [
#     "loan_amnt","term_months","int_rate","installment","annual_inc","dti",
#     "open_acc","pub_rec","inq_last_6mths","revol_bal","revol_util","total_acc","emp_length_yrs"
# ]
# CATEG = ["home_ownership","verification_status","purpose","application_type"]
# ALL = NUMERIC + CATEG

# # --- Sidebar controls ---
# thr = st.sidebar.slider("Decision threshold (≥ → Default)", 0.05, 0.95, float(thr_default), 0.01)

# # --- Input form ---
# st.subheader("Borrower & Loan Details")
# c1, c2 = st.columns(2)
# with c1:
#     loan_amnt   = st.number_input("Loan amount (USD)", 500, 200_000, 6_000, step=100)
#     term_months = st.selectbox("Term (months)", [36, 60], index=0)
#     int_rate    = st.number_input("Interest rate (%)", 0.0, 60.0, 18.0, step=0.1)
#     installment = st.number_input("Monthly installment", 1.0, 5_000.0, 220.0, step=1.0)
#     annual_inc  = st.number_input("Annual income", 0.0, 5_000_000.0, 120_000.0, step=1000.0)
#     dti         = st.number_input("DTI", 0.0, 200.0, 25.0, step=0.1)
# with c2:
#     open_acc       = st.number_input("Open credit lines", 0, 200, 12)
#     pub_rec        = st.number_input("Public derogatories", 0, 50, 0)
#     inq_last_6mths = st.number_input("Inquiries (6 months)", 0, 50, 1)
#     revol_bal      = st.number_input("Revolving balance", 0.0, 5_000_000.0, 8_000.0, step=100.0)
#     revol_util     = st.number_input("Revolving util (%)", 0.0, 300.0, 35.0, step=0.1)
#     total_acc      = st.number_input("Total credit lines", 0, 300, 30)
# emp_length_yrs = st.slider("Employment length (years)", 0, 10, 5)

# c3, c4 = st.columns(2)
# with c3:
#     home_ownership = st.selectbox("Home ownership", ["MORTGAGE","RENT","OWN","OTHER"])
#     verification_status = st.selectbox("Verification status", ["Verified","Source Verified","Not Verified"])
# with c4:
#     application_type = st.selectbox("Application type", ["INDIVIDUAL","JOINT"])
#     purpose = st.selectbox("Purpose", [
#         "debt_consolidation","credit_card","home_improvement","major_purchase","small_business","car",
#         "medical","wedding","house","moving","vacation","educational","other"
#     ])

# X_user = pd.DataFrame([{
#     "loan_amnt": loan_amnt, "term_months": term_months, "int_rate": int_rate,
#     "installment": installment, "annual_inc": annual_inc, "dti": dti,
#     "open_acc": open_acc, "pub_rec": pub_rec, "inq_last_6mths": inq_last_6mths,
#     "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
#     "emp_length_yrs": emp_length_yrs, "home_ownership": home_ownership,
#     "verification_status": verification_status, "purpose": purpose, "application_type": application_type
# }])[ALL]

# pred_tab, shap_tab = st.tabs(["🔮 Predict", "📊 SHAP (optional)"])

# with pred_tab:
#     if st.button("Predict with RF"):
#         proba = float(pipe.predict_proba(X_user)[:,1][0])
#         label = int(proba >= thr)
#         st.metric("Default probability (PD)", f"{proba:.2%}")
#         risk_score = (1.0 - proba) * 100.0
#         st.metric("Risk score (0–100)", f"{risk_score:.1f}")
#         if label == 1:
#             st.error(f"⚠️ Predicted: Default • thr={thr:.2f}")
#         else:
#             st.success(f"✅ Predicted: Fully Paid • thr={thr:.2f}")
#         with st.expander("Show inputs"):
#             st.write(X_user)


# with shap_tab:
#     st.caption("Global + local explanations via SHAP.")
#     try:
#         import shap, matplotlib.pyplot as plt
#         import numpy as np
#         import pandas as pd
#         from sklearn.linear_model import LogisticRegression
#         from sklearn.ensemble import RandomForestClassifier
#         try:
#             from xgboost import XGBClassifier
#         except Exception:
#             class XGBClassifier: ...  # dummy for isinstance check if xgboost isn't installed

#         if bg is None or bg.empty:
#             st.warning("No background (models/background.parquet).")
#         else:
#             # --- 1) prep transform ---
#             prep = pipe.named_steps["prep"]
#             clf  = pipe.named_steps["clf"]

#             X_bg   = prep.transform(bg[ALL])          # background in model space
#             X_one  = prep.transform(X_user[ALL])      # single input in model space
#             try:
#                 feat_names = prep.get_feature_names_out()
#             except Exception:
#                 feat_names = [f"f{i}" for i in range(X_bg.shape[1])]

#             # --- 2) choose the right explainer ---
#             if isinstance(clf, (RandomForestClassifier, XGBClassifier)):
#                 explainer = shap.TreeExplainer(clf)
#                 sv_bg  = explainer.shap_values(X_bg)
#                 sv_one = explainer.shap_values(X_one)
#                 # TreeExplainer returns list for multiclass; use class-1
#                 values_bg  = sv_bg[1] if isinstance(sv_bg, list)  else sv_bg
#                 values_one = sv_one[1] if isinstance(sv_one, list) else sv_one
#                 base_val   = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

#             elif isinstance(clf, LogisticRegression):
#                 explainer  = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")
#                 values_bg  = explainer.shap_values(X_bg)
#                 values_one = explainer.shap_values(X_one)
#                 base_val   = explainer.expected_value

#             else:
#                 # Fallback: KernelExplainer on the pipeline's predict_proba
#                 f = lambda A: pipe.predict_proba(pd.DataFrame(A, columns=ALL))[:, 1]
#                 explainer  = shap.Explainer(f, bg[ALL])
#                 ex_bg      = explainer(bg[ALL])
#                 ex_one     = explainer(X_user[ALL])
#                 values_bg, values_one = ex_bg.values, ex_one.values
#                 base_val   = explainer.expected_value

#             # --- 3) GLOBAL summary ---
#             fig1 = plt.figure()
#             shap.summary_plot(values_bg, features=X_bg, feature_names=feat_names, show=False)
#             st.pyplot(fig1, clear_figure=True)

#             # --- 4) LOCAL explanation (bar) ---
#             exp_one = shap.Explanation(
#                 values=values_one[0],
#                 base_values=base_val,
#                 data=X_one[0] if hasattr(X_one, "__getitem__") else X_one,
#                 feature_names=feat_names
#             )
#             fig2 = shap.plots.bar(exp_one, show=False)
#             st.pyplot(fig2, clear_figure=True)

#     except Exception as e:
#         st.error("SHAP failed. See details below, then we’ll fix the env/code as needed.")
#         st.exception(e)


# # with shap_tab:
# #     try:
# #         import shap, matplotlib.pyplot as plt
# #         if bg is None or bg.empty:
# #             st.info("Run notebook step to save models/background.parquet for SHAP.")
# #         else:
# #             @st.cache_resource
# #             def get_explainer():
# #                 return shap.Explainer(pipe, bg)
# #             explainer = get_explainer()
# #             st.write("Global importance (mean |SHAP|):")
# #             sv_bg = explainer(bg.sample(min(200, len(bg)), random_state=0))
# #             fig1 = shap.plots.bar(sv_bg, show=False)
# #             st.pyplot(fig1, clear_figure=True)
# #             st.write("This case:")
# #             sv_one = explainer(X_user)
# #             fig2 = shap.plots.bar(sv_one[0], show=False)
# #             st.pyplot(fig2, clear_figure=True)
# #     except Exception as e:
# #         st.warning("SHAP couldn’t run here. Check that `shap` is installed and background.parquet exists.")
# #         st.exception(e)



# XGBoost – Loan Default Predictor (with SHAP + LIME)
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from streamlit_shap import st_shap
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


st.set_page_config(page_title="Loan Default – XGBoost", page_icon="🤖", layout="wide")
st.title("XGBoost – Loan Default Predictor")

# ---------- Load artifacts ----------
@st.cache_resource(show_spinner=False)
def load_pipeline(path="models/xgb_pipeline.pkl"):
    # This is your saved XGB pipeline: Pipeline(preprocess, model)
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_background(path="models/bg_sample.parquet"):
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

@st.cache_resource(show_spinner=False)
def load_threshold(path="models/threshold.json", default=0.53):
    try:
        with open(path) as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return float(default)

pipe = load_pipeline()
bg   = load_background()
thr  = load_threshold()

# ---------- Schema (must match training) ----------
NUMERIC = [
    "loan_amnt","term_months","int_rate","installment","annual_inc","dti",
    "open_acc","pub_rec","inq_last_6mths","revol_bal","revol_util","total_acc","emp_length_yrs"
]
CATEG  = ["home_ownership","verification_status","purpose","application_type"]
ALL    = NUMERIC + CATEG

# ---------- Sidebar: threshold ----------
thr = st.sidebar.slider("Decision threshold (≥ → Default)", 0.05, 0.95, float(thr), 0.01)

# ---------- Input form ----------
st.subheader("Borrower & Loan Details")
c1, c2 = st.columns(2)
with c1:
    loan_amnt   = st.number_input("Loan amount (USD)", 500, 200_000, 6_000, step=100)
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
emp_length_yrs = st.slider("Employment length (years)", 0, 10, 5)

c3, c4 = st.columns(2)
with c3:
    home_ownership = st.selectbox("Home ownership", ["MORTGAGE","RENT","OWN","OTHER"])
    verification_status = st.selectbox("Verification status", ["Verified","Source Verified","Not Verified"])
with c4:
    application_type = st.selectbox("Application type", ["INDIVIDUAL","JOINT"])
    purpose = st.selectbox("Purpose", [
        "debt_consolidation","credit_card","home_improvement","major_purchase","small_business","car",
        "medical","wedding","house","moving","vacation","educational","other"
    ])

X_user = pd.DataFrame([{
    "loan_amnt": loan_amnt, "term_months": term_months, "int_rate": int_rate,
    "installment": installment, "annual_inc": annual_inc, "dti": dti,
    "open_acc": open_acc, "pub_rec": pub_rec, "inq_last_6mths": inq_last_6mths,
    "revol_bal": revol_bal, "revol_util": revol_util, "total_acc": total_acc,
    "emp_length_yrs": emp_length_yrs, "home_ownership": home_ownership,
    "verification_status": verification_status, "purpose": purpose, "application_type": application_type
}])[ALL]

tab_pred, tab_explain = st.tabs(["🔮 Predict", "🧠 Explain (SHAP & LIME)"])

# ---------- Predict ----------
with tab_pred:
    if st.button("Predict with XGBoost"):
        proba = float(pipe.predict_proba(X_user)[0,1])
        label = int(proba >= thr)

        st.metric("Default probability (PD)", f"{proba:.2%}")
        st.metric("Decision (thr={:.2f})".format(thr), "Default" if label==1 else "Fully Paid")
        with st.expander("Show inputs"):
            st.write(X_user)



# ---------- Explain ----------# ---------- Explain ----------
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

        
        # st.subheader("LIME — Local (top 10)")
        # lime_explainer = LimeTabularExplainer(
        #     training_data=bg[ALL].values,          # RAW background
        #     feature_names=ALL,
        #     class_names=["Fully Paid","Default"],
        #     mode='classification',
        #     discretize_continuous=True
        # )
        # lime_exp = lime_explainer.explain_instance(
        #     data_row=X_user.iloc[0].values,
        #     predict_fn=lambda X: pipe.predict_proba(pd.DataFrame(X, columns=ALL)),
        #     num_features=10
        # )
        # st.pyplot(lime_exp.as_pyplot_figure(), clear_figure=True)

        st.info(
            "Reading the charts:\n"
            "• SHAP beeswarm ranks features by global impact; right = higher default risk, left = lower.\n"
            "• SHAP waterfall explains this borrower: red bars raise risk, blue bars lower it.\n"
            "• LIME shows top local rules that support the decision."
        )
