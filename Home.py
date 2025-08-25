
# Home.py
import base64, time, os
from pathlib import Path
import pandas as pd
import streamlit as st

# Optional (only if you have models/)
try:
    import joblib
except Exception:
    joblib = None

st.set_page_config(page_title="NextGenCredit", layout="wide")

from pathlib import Path
import streamlit as st

st.markdown("A fast, friendly loan check. Enter a few details and get a clear yes/no — no impact on your credit score.")



HERE = Path(__file__).resolve().parent
APP_ROOT = HERE if HERE.name != "pages" else HERE.parent
css_file = APP_ROOT / "assets" / "css" / "styles.css"

if css_file.exists():
    st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS not found at: {css_file}")


# ---------- A) CSS (robust) ----------
HERE = Path(__file__).resolve().parent
APP_ROOT = HERE if HERE.name != "pages" else HERE.parent
CSS = APP_ROOT / "assets" / "css" / "styles.css"
if CSS.exists():
    st.markdown(f"<style>{CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
else:
    # Minimal fallback so you still get the font + brand color
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
      :root{--brand:#0c169a;--brand-dark:#080f6e;--hero-h:60vh}
      html,body,[class*="css"]{font-family:'Manrope',system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,'Helvetica Neue',Arial,sans-serif}
      .container{max-width:1180px;margin:0 auto;padding:0 1rem}
      .hero-frame{position:relative;height:var(--hero-h);border-radius:18px;overflow:hidden;display:flex;align-items:center;justify-content:center;background:linear-gradient(120deg,#101aa8 0%, #0c169a 60%, #08106b 100%)}
      .hero-svg-wrap{width:80%;height:80%;display:flex;align-items:center;justify-content:center}
      .hero-svg-wrap img{width:100%;height:100%;object-fit:contain;display:block}
      .section{padding:1.8rem 0}
      .mini-card{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:1rem}
      .pill{display:inline-block;border-radius:999px;padding:.25rem .6rem;font-weight:800}
      .pill.ok{background:#16a34a20;color:#166534;border:1px solid #16a34a55}
      .pill.no{background:#dc262620;color:#991b1b;border:1px solid #dc262655}
      .grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem}
      @media (max-width:980px){:root{--hero-h:46vh}.grid-3{grid-template-columns:1fr}}
      h1.title{margin:0 0 .8rem;font-weight:800}
      .muted{color:#475569}
      .stButton>button{background:var(--brand)!important;color:#fff!important;border:none!important;border-radius:12px!important;font-weight:800!important}
      .stButton>button:hover{background:var(--brand-dark)!important}
    </style>
    """, unsafe_allow_html=True)



# ---------- B) (Optional) Inject SVG sprite if you use <use href="#..."> icons ----------
SPRITE = APP_ROOT / "assets" / "svg" / "illustrations.svg"
if SPRITE.exists():
    st.markdown(SPRITE.read_text(encoding="utf-8"), unsafe_allow_html=True)



# =========================================================
# 1) HERO — rotating hero-*.svg (80% width & height inside a fixed frame)
# =========================================================
# --- HERO (display only; no sleep/rerun here) ---
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_ROOT = HERE if HERE.name != "pages" else HERE.parent
IMG_DIR = APP_ROOT / "assets" / "img"
SVG_FILES = sorted(IMG_DIR.glob("hero-*.svg"))

# init index & toggle once
st.session_state.setdefault("hero_idx", 0)
st.session_state.setdefault("animate_hero", True)  # you can set False to pause

def _show_svg(path: Path, container):
    svg_txt = path.read_text(encoding="utf-8")
    b64 = base64.b64encode(svg_txt.encode("utf-8")).decode("utf-8")
    container.markdown(
        '<div class="hero-svg-wrap">'
        f'<img src="data:image/svg+xml;base64,{b64}" alt="hero" />'
        '</div>',
        unsafe_allow_html=True
    )

st.markdown('<div class="hero-frame">', unsafe_allow_html=True)
slot = st.empty()
if SVG_FILES:
    _show_svg(SVG_FILES[st.session_state.hero_idx], slot)
else:
    st.info(f"Add SVGs like {IMG_DIR/'hero-1.svg'} to see the hero.")
st.markdown('</div>', unsafe_allow_html=True)

# Optional tiny control (remove if you don’t want it)
# st.toggle("🎞️ Animate hero", key="animate_hero", value=st.session_state.animate_hero)

# =========================================================
# 2) Simple loan prediction (3 inputs → Eligible / Not eligible)
# =========================================================


# --- Minimal bank-style eligibility (3 inputs, plain verdict) ---
st.markdown("### Can I get the loan?")
st.caption("Quick check using typical bank cut-offs for affordability (DTI) and loan-to-income. Indicative only.")

c1, c2, c3 = st.columns(3)
with c1:
    loan_amnt = st.number_input("Loan amount", min_value=1000, max_value=500_000, value=25_000, step=1000, key="min_loan")
with c2:
    annual_inc = st.number_input("Annual income", min_value=0, max_value=5_000_000, value=120_000, step=1000, key="min_income")
with c3:
    term_months = st.selectbox("Term (months)", options=[12, 24, 36, 48, 60], index=2, key="min_term")

if st.button("Check now", type="primary", key="min_check_btn"):
    # --- simple policy logic (no numbers shown to user) ---
    monthly_income = annual_inc / 12.0 if annual_inc else 0.0
    # rough installment with 5% loading to be conservative
    installment_est = (loan_amnt / max(int(term_months), 1)) * 1.05 if term_months else 0.0
    dti_pct = (installment_est / max(monthly_income, 1e-6)) * 100 if monthly_income else 100.0
    lti = loan_amnt / max(annual_inc, 1e-6) if annual_inc else 1.0

    # Typical bank-style cut-offs (keep it simple)
    # - Not eligible if DTI >= 40% OR Loan-to-income >= 60%
    # - Otherwise likely eligible
    reasons = []
    if dti_pct >= 40:
        reasons.append("affordability is tight")
    if lti >= 0.60:
        reasons.append("requested amount is high relative to income")

    if reasons:
        st.error("Not eligible (indicative)")
        st.caption("Reason: " + " and ".join(reasons) + ".")
    else:
        st.success("✅ Likely eligible (indicative)")
        st.caption("Within typical limits lenders look for.")




# ... rest of your widgets ...

# --- Footer ---
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #f9f9f9;
        text-align: center;
        padding: 8px;
        font-size: 13px;
        color: #666;
        border-top: 1px solid #ddd;
    }
    </style>
    <div class="footer">
        © 2025 NextGenCred | PocketLand Built with using Streamlit
    </div>
""", unsafe_allow_html=True)


# --- TIMER: rotate after everything is rendered ---
import time
if st.session_state.get("animate_hero", True) and SVG_FILES:
    time.sleep(3)
    st.session_state.hero_idx = (st.session_state.hero_idx + 1) % len(SVG_FILES)
    try:
        st.rerun()              # Streamlit ≥ 1.30
    except Exception:
        st.experimental_rerun() # older versions



