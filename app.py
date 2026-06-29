import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
}

/* Cards */
.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 12px;
}

.metric-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    color: #8b949e;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-card h2 {
    font-size: 28px !important;
    font-weight: 800 !important;
    margin: 0 !important;
}

/* Result boxes */
.result-low {
    background: linear-gradient(135deg, #0d2818, #1a3a2a);
    border: 2px solid #2ea043;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.result-medium {
    background: linear-gradient(135deg, #2d1f00, #3d2e00);
    border: 2px solid #d29922;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.result-high {
    background: linear-gradient(135deg, #2d0f0f, #3d1515);
    border: 2px solid #f85149;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    margin: 0 0 8px 0;
}

.result-prob {
    font-size: 48px;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    margin: 12px 0;
}

.result-desc {
    font-size: 14px;
    color: #8b949e;
    margin: 0;
}

/* Case scenario cards */
.case-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 18px;


.case-best  { border-left: 4px solid #2ea043; }
.case-avg   { border-left: 4px solid #d29922; }
.case-worst { border-left: 4px solid #f85149; }

.case-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    margin: 0 0 4px 0;
}

.case-desc {
    font-size: 12px;
    color: #8b949e;
    margin: 0;
}

/* Progress bar */
.prob-bar-wrap {
    background: #21262d;
    border-radius: 8px;
    height: 12px;
    margin: 16px 0;
    overflow: hidden;
}

.prob-bar-fill-low    { background: linear-gradient(90deg, #2ea043, #56d364); height: 100%; border-radius: 8px; transition: width 0.5s; }
.prob-bar-fill-medium { background: linear-gradient(90deg, #d29922, #e3b341); height: 100%; border-radius: 8px; transition: width 0.5s; }
.prob-bar-fill-high   { background: linear-gradient(90deg, #f85149, #ff7b72); height: 100%; border-radius: 8px; transition: width 0.5s; }

/* Section header */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: #1f2d3d;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    font-size: 11px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 16px;
}

/* Divider */
hr { border-color: #21262d !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #0d419d) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.85 !important; }

/* Inputs */
.stNumberInput input, .stSelectbox select {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 8px !important;
}

/* Sliders */
.stSlider .stSlider-track { background: #21262d !important; }

</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    model   = pickle.load(open(os.path.join(models_dir, 'loan_model.pkl'),      'rb'))
    scaler  = pickle.load(open(os.path.join(models_dir, 'scaler.pkl'),          'rb'))
    columns = pickle.load(open(os.path.join(models_dir, 'feature_columns.pkl'),'rb'))
    return model, scaler, columns

try:
    model, scaler, feature_columns = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model not found: {e}. Run train_model.py first.")


# ── Scenario presets ────────────────────────────────────────────────────────────
SCENARIOS = {
    "🟢 Best Case — Ideal Applicant": {
        "desc": "High income, excellent credit, low debt",
        "loan_amnt": 8000,
        "annual_inc": 120000,
        "dti": 5.0,
        "credit_score": 780,
        "emp_length": 10,
        "int_rate": 6.5,
        "installment": 240,
        "open_acc": 6,
        "revol_util": 10.0,
        "revol_bal": 2000,
        "total_acc": 18,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "pub_rec": 0,
        "pub_rec_bankruptcies": 0,
        "mort_acc": 1,
        "num_actv_bc_tl": 3,
        "bc_util": 8.0,
        "pct_tl_nvr_dlq": 98.0,
        "num_tl_90g_dpd_24m": 0,
        "term": 36,
        "home_ownership": "MORTGAGE",
        "purpose": "debt_consolidation",
        "grade": "A",
        "sub_grade": "A1",
        "verification_status": "Verified",
    },
    "🟡 Average Case — Typical Applicant": {
        "desc": "Middle income, average credit, moderate debt",
        "loan_amnt": 15000,
        "annual_inc": 60000,
        "dti": 18.0,
        "credit_score": 690,
        "emp_length": 4,
        "int_rate": 13.5,
        "installment": 510,
        "open_acc": 9,
        "revol_util": 45.0,
        "revol_bal": 8000,
        "total_acc": 22,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "pub_rec": 0,
        "pub_rec_bankruptcies": 0,
        "mort_acc": 0,
        "num_actv_bc_tl": 4,
        "bc_util": 45.0,
        "pct_tl_nvr_dlq": 85.0,
        "num_tl_90g_dpd_24m": 0,
        "term": 36,
        "home_ownership": "RENT",
        "purpose": "credit_card",
        "grade": "C",
        "sub_grade": "C3",
        "verification_status": "Source Verified",
    },
    "🔴 Worst Case — High Risk Applicant": {
        "desc": "Low income, poor credit, high debt, past defaults",
        "loan_amnt": 30000,
        "annual_inc": 28000,
        "dti": 38.0,
        "credit_score": 620,
        "emp_length": 0,
        "int_rate": 26.0,
        "installment": 950,
        "open_acc": 14,
        "revol_util": 88.0,
        "revol_bal": 22000,
        "total_acc": 28,
        "delinq_2yrs": 3,
        "inq_last_6mths": 5,
        "pub_rec": 1,
        "pub_rec_bankruptcies": 1,
        "mort_acc": 0,
        "num_actv_bc_tl": 8,
        "bc_util": 90.0,
        "pct_tl_nvr_dlq": 55.0,
        "num_tl_90g_dpd_24m": 2,
        "term": 60,
        "home_ownership": "RENT",
        "purpose": "small_business",
        "grade": "F",
        "sub_grade": "F4",
        "verification_status": "Not Verified",
    },
}


# ── Helper: build feature vector ────────────────────────────────────────────────
def build_features(vals):
    grades    = ['A','B','C','D','E','F','G']
    subgrades = [f'{g}{n}' for g in grades for n in range(1,6)]
    sg_map    = {sg: i+1 for i, sg in enumerate(subgrades)}

    home_map   = {'RENT':0,'OWN':1,'MORTGAGE':2,'ANY':3,'NONE':4}
    purpose_map = {
        'debt_consolidation':0,'credit_card':1,'home_improvement':2,
        'other':3,'major_purchase':4,'small_business':5,'car':6,
        'medical':7,'moving':8,'vacation':9,'house':10,
        'wedding':11,'renewable_energy':12,'educational':13
    }
    grade_map  = {g: i for i, g in enumerate(grades)}
    verif_map  = {'Not Verified':0,'Source Verified':1,'Verified':2}

    cs = vals['credit_score']

    row = {
        'loan_amnt'           : vals['loan_amnt'],
        'int_rate'            : vals['int_rate'],
        'installment'         : vals['installment'],
        'annual_inc'          : vals['annual_inc'],
        'dti'                 : vals['dti'],
        'open_acc'            : vals['open_acc'],
        'revol_bal'           : vals['revol_bal'],
        'revol_util'          : vals['revol_util'],
        'total_acc'           : vals['total_acc'],
        'emp_length'          : vals['emp_length'],
        'delinq_2yrs'         : vals['delinq_2yrs'],
        'inq_last_6mths'      : vals['inq_last_6mths'],
        'pub_rec'             : vals['pub_rec'],
        'pub_rec_bankruptcies': vals['pub_rec_bankruptcies'],
        'mort_acc'            : vals['mort_acc'],
        'num_actv_bc_tl'      : vals['num_actv_bc_tl'],
        'bc_util'             : vals['bc_util'],
        'pct_tl_nvr_dlq'      : vals['pct_tl_nvr_dlq'],
        'num_tl_90g_dpd_24m'  : vals['num_tl_90g_dpd_24m'],
        'credit_score'        : cs,
        'sub_grade'           : sg_map.get(vals['sub_grade'], 10),
        'home_ownership'      : home_map.get(vals['home_ownership'], 0),
        'purpose'             : purpose_map.get(vals['purpose'], 3),
        'grade'               : grade_map.get(vals['grade'], 2),
        'verification_status' : verif_map.get(vals['verification_status'], 0),
        'term'                : vals['term'],
        # Engineered
        'monthly_debt_to_income': vals['installment'] / (vals['annual_inc']/12 + 1),
        'revol_to_income'       : vals['revol_bal']   / (vals['annual_inc'] + 1),
        'loan_to_income'        : vals['loan_amnt']   / (vals['annual_inc'] + 1),
        'util_pressure'         : vals['revol_util']  * vals['bc_util'] / 100,
        'score_util_ratio'      : cs / (vals['revol_util'] + 1),
        'dti_rate_stress'       : vals['dti'] * vals['int_rate'],
        'score_dti_risk'        : cs / (vals['dti'] + 1),
        'delinq_risk'           : (vals['delinq_2yrs']*2 + vals['pub_rec']*3 +
                                   vals['pub_rec_bankruptcies']*5 +
                                   vals['num_tl_90g_dpd_24m']*4),
        'open_acc_ratio'        : vals['open_acc'] / (vals['total_acc'] + 1),
        'grade_score_gap'       : sg_map.get(vals['sub_grade'],10)*10 / (cs-500+1),
        'total_loan_cost'       : vals['installment'] * vals['term'],
        'cost_to_income'        : (vals['installment']*vals['term']) / (vals['annual_inc']+1),
    }

    df_in = pd.DataFrame([row])

    # Align to training columns
    for col in feature_columns:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[feature_columns]

    return df_in


# ── Predict helper ──────────────────────────────────────────────────────────────
def predict(vals):
    df_in  = build_features(vals)
    scaled = scaler.transform(df_in)
    prob   = float(model.predict_proba(scaled)[0][1])
    return prob


# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:32px 0 8px 0">
<h1 style="font-size:40px;font-weight:800;margin:0 0 6px 0;color:#e6edf3">
        🏦 Loan Default Predictor
    </h1>

</div>
<hr>
""", unsafe_allow_html=True)


# ── Scenario buttons ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Quick Scenario Presets</div>',
            unsafe_allow_html=True)

col_b, col_a, col_w = st.columns(3)

scenario_selected = None
with col_b:
    st.markdown("""<div class="case-card case-best">
        <div class="case-title" style="color:#2ea043">🟢 Best Case</div>
        <div class="case-desc">High income · Excellent credit · Low debt</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Load Best Case", key="btn_best"):
        scenario_selected = "🟢 Best Case — Ideal Applicant"

with col_a:
    st.markdown("""<div class="case-card case-avg">
        <div class="case-title" style="color:#d29922">🟡 Average Case</div>
        <div class="case-desc">Middle income · Average credit · Moderate debt</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Load Average Case", key="btn_avg"):
        scenario_selected = "🟡 Average Case — Typical Applicant"

with col_w:
    st.markdown("""<div class="case-card case-worst">
        <div class="case-title" style="color:#f85149">🔴 Worst Case</div>
        <div class="case-desc">Low income · Poor credit · Past defaults</div>
    </div>""", unsafe_allow_html=True)
    if st.button("Load Worst Case", key="btn_worst"):
        scenario_selected = "🔴 Worst Case — High Risk Applicant"

# Apply scenario to session state
if scenario_selected:
    st.session_state['scenario'] = SCENARIOS[scenario_selected]

defaults = st.session_state.get('scenario', SCENARIOS["🟡 Average Case — Typical Applicant"])

st.markdown("<hr>", unsafe_allow_html=True)


# ── Input form ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Applicant Details</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**💰 Loan & Income**")
    loan_amnt  = st.number_input("Loan Amount ($)",
                     500, 40000, int(defaults['loan_amnt']), step=500)
    annual_inc = st.number_input("Annual Income ($)",
                     10000, 500000, int(defaults['annual_inc']), step=1000)
    installment = st.number_input("Monthly Installment ($)",
                     50, 3000, int(defaults['installment']), step=10)
    int_rate   = st.slider("Interest Rate (%)",
                     5.0, 31.0, float(defaults['int_rate']), step=0.5)
    term       = st.selectbox("Loan Term (months)",
                     [36, 60],
                     index=0 if defaults['term']==36 else 1)

with c2:
    st.markdown("**📊 Credit Profile**")
    credit_score = st.slider("Credit Score (FICO)",
                     580, 850, int(defaults['credit_score']))
    dti          = st.slider("Debt-to-Income Ratio (%)",
                     0.0, 50.0, float(defaults['dti']), step=0.5)
    revol_util   = st.slider("Revolving Utilization (%)",
                     0.0, 100.0, float(defaults['revol_util']), step=1.0)
    bc_util      = st.slider("Bankcard Utilization (%)",
                     0.0, 100.0, float(defaults['bc_util']), step=1.0)
    pct_tl_nvr_dlq = st.slider("% Tradelines Never Delinquent",
                     0.0, 100.0, float(defaults['pct_tl_nvr_dlq']), step=1.0)

with c3:
    st.markdown("**📋 History & Employment**")
    emp_length   = st.slider("Employment Length (years)",
                     0, 10, int(defaults['emp_length']))
    delinq_2yrs  = st.number_input("Delinquencies (past 2 yrs)",
                     0, 20, int(defaults['delinq_2yrs']))
    inq_last_6mths = st.number_input("Credit Inquiries (past 6 months)",
                     0, 20, int(defaults['inq_last_6mths']))
    pub_rec      = st.number_input("Public Records",
                     0, 10, int(defaults['pub_rec']))
    pub_rec_bankruptcies = st.number_input("Bankruptcies",
                     0, 5, int(defaults['pub_rec_bankruptcies']))

# Row 2
c4, c5, c6 = st.columns(3)

with c4:
    st.markdown("**🏠 Property & Purpose**")
    home_ownership = st.selectbox("Home Ownership",
                     ["RENT","OWN","MORTGAGE","ANY","NONE"],
                     index=["RENT","OWN","MORTGAGE","ANY","NONE"]
                           .index(defaults['home_ownership']))
    purpose = st.selectbox("Loan Purpose",
                ["debt_consolidation","credit_card","home_improvement",
                 "other","major_purchase","small_business","car",
                 "medical","moving","vacation","house",
                 "wedding","renewable_energy","educational"],
                index=["debt_consolidation","credit_card","home_improvement",
                       "other","major_purchase","small_business","car",
                       "medical","moving","vacation","house",
                       "wedding","renewable_energy","educational"]
                      .index(defaults['purpose']))

with c5:
    st.markdown("**🏷️ Loan Grade**")
    grade = st.selectbox("LC Grade",
                ["A","B","C","D","E","F","G"],
                index=["A","B","C","D","E","F","G"].index(defaults['grade']))
    all_subs = [f'{g}{n}' for g in ["A","B","C","D","E","F","G"]
                           for n in range(1,6)]
    sub_grade = st.selectbox("Sub Grade", all_subs,
                index=all_subs.index(defaults['sub_grade']))
    verification_status = st.selectbox("Verification Status",
                ["Not Verified","Source Verified","Verified"],
                index=["Not Verified","Source Verified","Verified"]
                      .index(defaults['verification_status']))

with c6:
    st.markdown("**💳 Credit Lines**")
    open_acc   = st.number_input("Open Credit Lines",
                     1, 60, int(defaults['open_acc']))
    total_acc  = st.number_input("Total Credit Lines",
                     1, 100, int(defaults['total_acc']))
    revol_bal  = st.number_input("Revolving Balance ($)",
                     0, 200000, int(defaults['revol_bal']), step=500)
    mort_acc   = st.number_input("Mortgage Accounts",
                     0, 20, int(defaults['mort_acc']))
    num_actv_bc_tl = st.number_input("Active Bankcard Tradelines",
                     0, 30, int(defaults['num_actv_bc_tl']))
    num_tl_90g_dpd_24m = st.number_input("Tradelines 90+ DPD (24m)",
                     0, 20, int(defaults['num_tl_90g_dpd_24m']))


# ── Predict button ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Default Risk", use_container_width=True)

if predict_btn and model_loaded:

    vals = dict(
        loan_amnt=loan_amnt, annual_inc=annual_inc, installment=installment,
        int_rate=int_rate, term=term, credit_score=credit_score, dti=dti,
        revol_util=revol_util, bc_util=bc_util, pct_tl_nvr_dlq=pct_tl_nvr_dlq,
        emp_length=emp_length, delinq_2yrs=delinq_2yrs,
        inq_last_6mths=inq_last_6mths, pub_rec=pub_rec,
        pub_rec_bankruptcies=pub_rec_bankruptcies, home_ownership=home_ownership,
        purpose=purpose, grade=grade, sub_grade=sub_grade,
        verification_status=verification_status, open_acc=open_acc,
        total_acc=total_acc, revol_bal=revol_bal, mort_acc=mort_acc,
        num_actv_bc_tl=num_actv_bc_tl, num_tl_90g_dpd_24m=num_tl_90g_dpd_24m,
    )

    prob = predict(vals)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Prediction Result</div>',
                unsafe_allow_html=True)

    # ── Result card ──────────────────────────────────────────────────────────────
    if prob < 0.25:
        rating     = "LOW RISK"
        css_class  = "result-low"
        bar_class  = "prob-bar-fill-low"
        color      = "#2ea043"
        emoji      = "✅"
        model_qual = "🏆 BEST"
        qual_color = "#2ea043"
        advice     = "Strong application. High likelihood of full repayment."
    elif prob < 0.45:
        rating     = "MODERATE RISK"
        css_class  = "result-medium"
        bar_class  = "prob-bar-fill-medium"
        color      = "#d29922"
        emoji      = "⚠️"
        model_qual = "👍 GOOD"
        qual_color = "#d29922"
        advice     = "Borderline application. Consider additional verification."
    elif prob < 0.65:
        rating     = "HIGH RISK"
        css_class  = "result-high"
        bar_class  = "prob-bar-fill-high"
        color      = "#f85149"
        emoji      = "🚨"
        model_qual = "⚠️ POOR"
        qual_color = "#f85149"
        advice     = "Elevated default risk. Recommend rejection or collateral."
    else:
        rating     = "VERY HIGH RISK"
        css_class  = "result-high"
        bar_class  = "prob-bar-fill-high"
        color      = "#f85149"
        emoji      = "🛑"
        model_qual = "❌ VERY POOR"
        qual_color = "#f85149"
        advice     = "Severe default risk. Strong recommendation to reject."

    bar_w = int(prob * 100)

    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title" style="color:{color}">{emoji} {rating}</div>
        <div class="result-prob" style="color:{color}">{prob:.1%}</div>
        <div class="prob-bar-wrap">
            <div class="{bar_class}" style="width:{bar_w}%"></div>
        </div>
        <div class="result-desc">{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key risk drivers ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Risk Drivers</div>',
                unsafe_allow_html=True)

    drivers = {
        "DTI × Rate Stress"      : round(dti * int_rate, 1),
        "Revolving Utilization"  : f"{revol_util:.0f}%",
        "Credit Score"           : credit_score,
        "Delinquency Risk Score" : delinq_2yrs*2 + pub_rec*3 + pub_rec_bankruptcies*5 + num_tl_90g_dpd_24m*4,
        "Loan to Income"         : f"{loan_amnt/annual_inc:.2f}x",
        "Monthly Debt/Income"    : f"{installment/(annual_inc/12):.1%}",
    }

    d_cols = st.columns(3)
    for i, (k, v) in enumerate(drivers.items()):
        with d_cols[i % 3]:
            st.metric(k, v)

    # ── Scenario comparison ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Scenario Comparison</div>',
                unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    for col, name, key in zip(
        [sc1, sc2, sc3],
        ["🟢 Best Case", "🟡 Average Case", "🔴 Worst Case"],
        ["🟢 Best Case — Ideal Applicant",
         "🟡 Average Case — Typical Applicant",
         "🔴 Worst Case — High Risk Applicant"]
    ):
        sp = predict(SCENARIOS[key])
        if sp < 0.25:   sc_color = "#2ea043"; sc_label = "LOW RISK"
        elif sp < 0.45: sc_color = "#d29922"; sc_label = "MODERATE"
        elif sp < 0.65: sc_color = "#f85149"; sc_label = "HIGH RISK"
        else:           sc_color = "#f85149"; sc_label = "VERY HIGH"

        with col:
            st.markdown(f"""<div class="metric-card">
                <h4>{name}</h4>
                <h2 style="color:{sc_color}">{sp:.1%}</h2>
                <div style="font-size:12px;color:{sc_color};
                            font-family:'Syne',sans-serif;
                            font-weight:700">{sc_label}</div>
            </div>""", unsafe_allow_html=True)


    perf_data = {
        "Metric"     : ["ROC-AUC","Recall (Default)","Precision (Default)",
                         "F1 (Default)","F1 (Weighted)","Training Samples"],
        "Value"      : ["0.7273","68.41%","36.07%","47.30%","69.92%","498,145"],
        "Rating"     : ["⭐⭐⭐ Good","⭐⭐⭐ Good","⭐⭐ Average",
                         "⭐⭐ Average","⭐⭐⭐ Good","—"],
        "Benchmark"  : ["> 0.70 = Good","> 60% = Good","> 50% = Good",
                         "> 50% = Good","> 65% = Good","600k+ rows"],
    }

    st.dataframe(
        pd.DataFrame(perf_data),
        use_container_width=True,
        hide_index=True
    )




# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:20px;
                font-weight:800;color:#e6edf3;margin-bottom:4px">
        🏦 About This Model
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="border-color:#21262d;margin:12px 0">
    """, unsafe_allow_html=True)

    st.markdown("""
    **Dataset**
    Lending Club · Real loans 2007–2018
    622,682 resolved loan records

    **Algorithm**
    Gradient Boosting Classifier
    300 estimators · max depth 5
    Balanced class weights

    **Features**
    33 features including 12 engineered
    interaction terms
    """)



    st.markdown("<hr style='border-color:#21262d'>", unsafe_allow_html=True)

    st.markdown("""
    **Risk Thresholds**
    """)

    thresholds_info = [
        ("#2ea043", "< 25%",    "LOW RISK — Approve"),
        ("#d29922", "25–45%",   "MODERATE — Review"),
        ("#f85149", "45–65%",   "HIGH RISK — Caution"),
        ("#f85149", "> 65%",    "VERY HIGH — Reject"),
    ]
    for color, rng, label in thresholds_info:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;
                    padding:5px 0;font-size:12px">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{color};flex-shrink:0"></div>
            <span style="color:#8b949e;width:55px">{rng}</span>
            <span style="color:#e6edf3">{label}</span>
        </div>
        """, unsafe_allow_html=True)