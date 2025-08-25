# components/ui.py
import os, datetime
import streamlit as st

def inject_css():
    css_path = os.path.join("assets", "css", "styles.css")
    if os.path.exists(css_path):
        st.markdown(f"<style>{open(css_path, 'r', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

def navbar(active: str = "home", brand_name: str = "Business Loans"):
    # Top bar HTML
    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-inner">
        <div class="brand">
          <div class="logo"></div>
          <div class="text">{brand_name}</div>
        </div>
        <div class="nav">
          <a class="{'active' if active=='home' else ''}" href="#">Home</a>
          <a class="{'active' if active=='credit' else ''}" href="#credit">Credit scoring</a>
          <a href="#how">How it works</a>
          <a href="#faq">FAQ</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Functional page links (so navigation actually changes pages too)
    # Display as compact pill buttons beneath (hidden by CSS would be overkill; keep simple)
    cols = st.columns([1,1,1,1,8])
    with cols[0]:
        try:
            st.page_link("Home.py", label="Home", icon="🏠")
        except Exception:
            pass
    with cols[1]:
        try:
            st.page_link("pages/2_Credit_Scoring.py", label="Credit scoring", icon="📊")
        except Exception:
            pass
    with cols[2]:
        st.markdown('<div></div>', unsafe_allow_html=True)  # spacer

def footer(company: str = "Your Company", email: str = "support@example.com"):
    year = datetime.datetime.now().year
    st.markdown("""
    <div class="footer" id="footer">
      <div class="footer-inner">
        <div>
          <h4>About</h4>
          <p>Indicative eligibility only. Final decisions depend on lender policies, affordability checks and verification.</p>
        </div>
        <div>
          <h4>Resources</h4>
          <p><a href="#how">How it works</a></p>
          <p><a href="#credit">Credit scoring (analyst)</a></p>
          <p><a href="#faq">FAQ</a></p>
        </div>
        <div>
          <h4>Contact</h4>
          <p>Email: <a href="mailto:{email}">{email}</a></p>
          <p>Hours: Mon–Fri 9:00–17:00</p>
        </div>
      </div>
      <div class="footer-bottom">
        <div>© {year} {company}. All rights reserved.</div>
        <div><a href="#">Terms</a> · <a href="#">Privacy</a> · <a href="#footer">Back to top ↑</a></div>
      </div>
    </div>
    """.format(year=year, company=company, email=email), unsafe_allow_html=True)
