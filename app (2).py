import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Diabetic Retinopathy Prediction in Patients", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "dataset.csv"

FEATURE_COLS = ["age", "systolic_bp", "diastolic_bp", "cholesterol", "pulse_pressure"]
FEATURE_LABELS = ["Age", "Systolic BP", "Diastolic BP", "Cholesterol", "Pulse Pressure"]

FALLBACK_SCALER_MEAN = np.array([60.45734923178542, 100.7949801885875, 90.55802524988333,
                                  100.72212041283959, 10.236954938704166])
FALLBACK_SCALER_SCALE = np.array([8.560073476802858, 10.66937583849836, 9.695041693723688,
                                   10.435460888632315, 11.175645820159119])
FALLBACK_LR_COEF = np.array([1.0713804792196202, 0.4693719901450734, 0.23190006732024973,
                              0.6045143861224852, 0.24693206941979687])
FALLBACK_LR_INTERCEPT = 0.12073183008385849

FALLBACK_RESULTS = pd.DataFrame([
    {"Model": "SVM", "Accuracy": 0.7708, "F1": 0.7788, "AUC": 0.8361},
    {"Model": "Logistic Regression", "Accuracy": 0.7675, "F1": 0.7688, "AUC": 0.8415},
    {"Model": "Gradient Boosting", "Accuracy": 0.7533, "F1": 0.7617, "AUC": 0.8408},
    {"Model": "Naive Bayes", "Accuracy": 0.7408, "F1": 0.7330, "AUC": 0.8326},
    {"Model": "Random Forest", "Accuracy": 0.7350, "F1": 0.7419, "AUC": 0.8081},
    {"Model": "KNN", "Accuracy": 0.7158, "F1": 0.7239, "AUC": 0.7791},
    {"Model": "Decision Tree", "Accuracy": 0.6750, "F1": 0.6793, "AUC": 0.6751},
])

FALLBACK_GROUP_MEANS = {
    "healthy": {"age": 57.14, "systolic_bp": 96.96, "diastolic_bp": 88.70, "cholesterol": 97.24},
    "risk": {"age": 63.60, "systolic_bp": 104.22, "diastolic_bp": 92.21, "cholesterol": 103.83},
}

RANGES = {
    "age": (30.0, 95.0),
    "systolic_bp": (70.0, 150.0),
    "diastolic_bp": (60.0, 130.0),
    "cholesterol": (70.0, 150.0),
}


def dataset_exists():
    return os.path.exists(DATA_PATH)


def load_dataset():
    return pd.read_csv(DATA_PATH) if dataset_exists() else None


@st.cache_resource(show_spinner="Training models on dataset.csv ...")
def load_model():
    if not dataset_exists():
        return {
            "scaler_mean": FALLBACK_SCALER_MEAN,
            "scaler_scale": FALLBACK_SCALER_SCALE,
            "lr_coef": FALLBACK_LR_COEF,
            "lr_intercept": FALLBACK_LR_INTERCEPT,
            "results": FALLBACK_RESULTS,
            "group_means": FALLBACK_GROUP_MEANS,
            "sample": None,
            "live": False,
        }

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH).drop(columns=["ID"], errors="ignore")
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    y = LabelEncoder().fit_transform(df["prognosis"])
    X = df[FEATURE_COLS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models = [
        ("Logistic Regression", LogisticRegression(random_state=42)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVM", SVC(random_state=42, probability=True)),
        ("Naive Bayes", GaussianNB()),
    ]

    rows = []
    for name, model in models:
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_s)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred), "F1": f1_score(y_test, pred), "AUC": roc_auc_score(y_test, prob)})

    results = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)
    rf = dict(models)["Random Forest"]
    rf_importance = pd.Series(rf.feature_importances_, index=FEATURE_COLS).rename({
        "age": "Age", "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP", "cholesterol": "Cholesterol", "pulse_pressure": "Pulse Pressure",
    }).sort_values(ascending=False)

    df["prognosis_label"] = LabelEncoder().fit_transform(df["prognosis"]).astype(object)
    df["prognosis_label"] = df["prognosis"].astype(str)
    group_means = {
        "healthy": df[df.prognosis == "no_retinopathy"][FEATURE_COLS[:4]].mean().to_dict(),
        "risk": df[df.prognosis == "retinopathy"][FEATURE_COLS[:4]].mean().to_dict(),
    }

    sample = pd.concat([
        df[df.prognosis == "no_retinopathy"].sample(min(220, len(df[df.prognosis == "no_retinopathy"])), random_state=42),
        df[df.prognosis == "retinopathy"].sample(min(220, len(df[df.prognosis == "retinopathy"])), random_state=42),
    ])[FEATURE_COLS[:4] + ["prognosis"]]
    sample = sample.rename(columns={"prognosis": "prognosis_label"})

    return {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "lr_coef": dict(models)["Logistic Regression"].coef_[0],
        "lr_intercept": dict(models)["Logistic Regression"].intercept_[0],
        "results": results,
        "rf_importance": rf_importance,
        "group_means": group_means,
        "sample": sample,
        "live": True,
    }


MODEL = load_model()
DATASET = load_dataset()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_risk(age, sbp, dbp, chol):
    pp = sbp - dbp
    raw = np.array([age, sbp, dbp, chol, pp])
    scaled = (raw - MODEL["scaler_mean"]) / MODEL["scaler_scale"]
    contributions = scaled * MODEL["lr_coef"]
    prob = sigmoid(MODEL["lr_intercept"] + contributions.sum())
    return prob, contributions, pp


def risk_band(prob):
    if prob < 0.35:
        return "LOW", "#2FA898"
    if prob < 0.65:
        return "BORDERLINE", "#E8B454"
    return "ELEVATED", "#D9502F"


def norm_pos(val, rng):
    lo, hi = rng
    return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100.0))


def panel(title, body=""):
    st.markdown(f'<div class="panel"><div class="eyebrow">{title}</div>{body}</div>', unsafe_allow_html=True)


def find_dataset_row(dataset, patient_id):
    if dataset is None or not str(patient_id).strip():
        return None
    key = str(patient_id).strip().lower()
    matches = dataset[dataset["ID"].astype(str).str.strip().str.lower() == key]
    return matches.iloc[0] if not matches.empty else None


def build_live_profile_summary(age, sbp, dbp, chol, contributions):
    healthy = MODEL["group_means"]["healthy"]
    risk = MODEL["group_means"]["risk"]
    current_vals = {"age": age, "systolic_bp": sbp, "diastolic_bp": dbp, "cholesterol": chol}
    labels = [("Age", "age"), ("Systolic BP", "systolic_bp"), ("Diastolic BP", "diastolic_bp"), ("Cholesterol", "cholesterol")]
    rows = []
    for name, key in labels:
        current = current_vals[key]
        healthy_mean = healthy[key]
        risk_mean = risk[key]
        diff = abs(current - healthy_mean)
        risk_label = "low health risk" if diff <= abs(current - risk_mean) else "high health risk"
        rows.append((name, current, healthy_mean, risk_mean, current - healthy_mean, risk_label))

    top_idx = int(np.argmax(np.abs(contributions)))
    top_label = FEATURE_LABELS[top_idx]
    top_text = "raises the score" if contributions[top_idx] >= 0 else "reduces the score"
    return rows, top_label, top_text


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
.stApp { background: radial-gradient(circle at 20% -10%, #10201A 0%, #070E0C 55%); color: #EDEAE1; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; }
.eyebrow { font-size: 11px; letter-spacing: 3px; color: #2FA898; font-weight: 700; margin-bottom: 6px; text-transform: uppercase; }
.panel { background: linear-gradient(160deg, #101C18, #0B1512); border: 1px solid rgba(232,180,84,0.10); border-radius: 16px; padding: 20px 22px; margin-bottom: 18px; }
.metric-mono { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

best_auc = MODEL["results"]["AUC"].max()
lr_acc = float(MODEL["results"].query('Model == "Logistic Regression"')["Accuracy"].iloc[0]) if not MODEL["results"].query('Model == "Logistic Regression"').empty else 0.7675
model_tag = "live-trained" if MODEL["live"] else "pretrained fallback"

cols = st.columns([2.2, 1])
with cols[0]:
    st.markdown('<div class="eyebrow">Ophthalmic screening model · logistic regression</div>', unsafe_allow_html=True)
    st.markdown("# Diabetic Retinopathy Prediction in Patients")
    st.markdown("This dashboard presents a simple screening workflow for diabetic retinopathy risk, using age, blood pressure, and cholesterol to estimate potential risk and compare the patient against a population reference.")
    st.markdown("### Home Page")
    st.markdown("Use the patient details section below to enter the available measurements and review an instant risk assessment, supporting analysis, and model performance summary.")
with cols[1]:
    st.markdown(
        f'<div style="text-align:right; margin-top:28px;">'
        f'<span class="badge">AUC {best_auc:.3f}</span>'
        f'<span class="badge">Accuracy {lr_acc*100:.1f}%</span>'
        f'<br><span style="font-size:11px;color:#5C7A70;">model: {model_tag}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

patient_id = st.text_input("Patient ID", placeholder="Enter patient ID")
matched_row = find_dataset_row(DATASET, patient_id)

defaults = {"age": 60.5, "systolic_bp": 101.0, "diastolic_bp": 90.5, "cholesterol": 101.0}
if matched_row is not None:
    defaults.update({k: float(matched_row[k]) for k in ["age", "systolic_bp", "diastolic_bp", "cholesterol"]})

col_in, col_gauge = st.columns([1, 1.4])
with col_in:
    panel("Patient Details", "<p>Enter the available clinical values below to generate a screening estimate.</p>")
    age = st.slider("Age (years)", *RANGES["age"], defaults["age"], 0.5)
    sbp = st.slider("Systolic BP (mmHg)", *RANGES["systolic_bp"], defaults["systolic_bp"], 0.5)
    dbp = st.slider("Diastolic BP (mmHg)", *RANGES["diastolic_bp"], defaults["diastolic_bp"], 0.5)
    chol = st.slider("Cholesterol (mg/dL)", *RANGES["cholesterol"], defaults["cholesterol"], 0.5)
    prob, contributions, pulse_pressure = compute_risk(age, sbp, dbp, chol)
    if matched_row is not None:
        actual = str(matched_row.get("prognosis", "Unknown")).replace("_", " ")
        st.markdown(
            f'<div style="margin-top:10px;padding:10px;border-radius:10px;background:rgba(47,168,152,0.10);border:1px solid rgba(47,168,152,0.25);font-size:12px;color:#EDEAE1;">'
            f'<strong>Matched dataset record</strong><br>'
            f'Age: <span class="metric-mono">{matched_row["age"]:.1f}</span> · '
            f'Systolic BP: <span class="metric-mono">{matched_row["systolic_bp"]:.1f}</span> · '
            f'Diastolic BP: <span class="metric-mono">{matched_row["diastolic_bp"]:.1f}</span><br>'
            f'Cholesterol: <span class="metric-mono">{matched_row["cholesterol"]:.1f}</span> · '
            f'Real outcome: <span class="metric-mono">{actual}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div style="margin-top:10px;font-size:12px;color:#5C7A70;">Derived pulse pressure: <span class="metric-mono" style="color:#EDEAE1;">{pulse_pressure:.1f} mmHg</span></div>', unsafe_allow_html=True)

with col_gauge:
    label, color = risk_band(prob)
    prediction_label = "Retinopathy" if prob >= 0.5 else "No Retinopathy"
    panel("Prediction", "<p>This estimate updates automatically as you adjust the clinical inputs.</p>")
    st.metric("Risk %", f"{prob * 100:.1f}%")
    st.metric("Prediction", prediction_label)
    st.markdown(
        f'<div style="margin-top:12px;padding:12px;border-radius:12px;background:rgba(232,180,84,0.10);border:1px solid rgba(232,180,84,0.25);">'
        f'<div style="font-size:12px;color:#9CA8A3;">Patient ID: <span class="metric-mono" style="color:#EDEAE1;">{patient_id or "Not provided"}</span></div>'
        f'<div style="font-size:12px;color:#9CA8A3;margin-top:6px;">Prediction: <span class="metric-mono" style="color:#EDEAE1;">{prediction_label}</span></div>'
        f'<div style="font-size:12px;color:#9CA8A3;margin-top:6px;">Risk Score: <span class="metric-mono" style="color:#EDEAE1;">{prob * 100:.1f}%</span></div>'
        f'<div style="font-size:12px;color:#9CA8A3;margin-top:6px;">Prediction Time: <span class="metric-mono" style="color:#EDEAE1;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 46, "color": "#EDEAE1", "family": "IBM Plex Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#5C7A70", "tickfont": {"color": "#9CA8A3"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 35], "color": "rgba(47,168,152,0.15)"},
                {"range": [35, 65], "color": "rgba(232,180,84,0.15)"},
                {"range": [65, 100], "color": "rgba(217,80,47,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.9, "value": prob * 100},
        },
    ))
    fig.update_layout(height=290, margin=dict(t=10, b=0, l=30, r=30), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#EDEAE1"})
    st.plotly_chart(fig, width="stretch")
    st.markdown(f'<div style="text-align:center; font-size:13px; letter-spacing:2px; font-weight:700; color:{color};">{label} RISK</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="margin-top:8px;text-align:center;">' +
        ''.join(
            f'<span style="margin-right:16px;font-size:12px;color:#9CA8A3;">'
            f'<span style="color:{"#D9502F" if c >= 0 else "#2FA898"};">●</span> {lab} '
            f'<span class="metric-mono" style="color:#EDEAE1;">{c:+.2f}</span></span>'
            for lab, c in zip(FEATURE_LABELS, contributions)
        ) +
        '</div>',
        unsafe_allow_html=True,
    )

col_features = st.columns([1])[0]
with col_features:
    rows, top_label, top_text = build_live_profile_summary(age, sbp, dbp, chol, contributions)
    panel("Risk Analysis", "<h4>Feature contributions and clinical context</h4>" + "".join(
        f'<div style="margin-bottom:10px;"><div style="display:flex;justify-content:space-between;font-size:12px;color:#9CA8A3;"><span>{label}</span><span class="metric-mono" style="color:#EDEAE1;">{current:.1f}</span></div><div style="font-size:11px;color:#5C7A70;margin-top:2px;">Healthy mean {healthy_mean:.1f} · Elevated-risk mean {risk_mean:.1f} · {risk_label}</div></div>'
        for label, current, healthy_mean, risk_mean, delta, risk_label in rows
    ) + f'<div style="margin-top:12px;padding:10px 12px;border-radius:12px;background:rgba(47,168,152,0.10);border:1px solid rgba(47,168,152,0.2);font-size:12px;color:#EDEAE1;"><span style="color:#2FA898;font-weight:700;">Most influential today:</span> {top_label} is the strongest contributor and {top_text}.</div>')
