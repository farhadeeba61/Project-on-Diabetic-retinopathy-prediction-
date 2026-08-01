import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetic Retinopathy Prediction in Patients", layout="wide")

# 1. MODEL — trained live on dataset.csv if present, else the
#    real coefficients from that same training run (fallback so
#    the app always works even without the CSV alongside it).
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

FALLBACK_RF_IMPORTANCE = pd.Series(
    {"Age": 0.296, "Cholesterol": 0.218, "Systolic BP": 0.210, "Diastolic BP": 0.139, "Pulse Pressure": 0.137}
)

FALLBACK_GROUP_MEANS = {
    "healthy": {"age": 57.14, "systolic_bp": 96.96, "diastolic_bp": 88.70, "cholesterol": 97.24},
    "risk": {"age": 63.60, "systolic_bp": 104.22, "diastolic_bp": 92.21, "cholesterol": 103.83},
}

RANGES = {
    "age": (30, 95),
    "systolic_bp": (70, 150),
    "diastolic_bp": (60, 130),
    "cholesterol": (70, 150),
}


@st.cache_resource(show_spinner="Training models on dataset.csv ...")
def load_model():
    """Train live on dataset.csv if it's next to this script; otherwise fall
    back to the parameters already trained on that file, so the app is never
    blocked on a missing CSV."""
    path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if not os.path.exists(path):
        path = "dataset.csv"

    if os.path.exists(path):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        df = pd.read_csv(path)
        df = df.drop(columns=["ID"], errors="ignore")
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

        le = LabelEncoder()
        y = le.fit_transform(df["prognosis"])  # 0 = no_retinopathy, 1 = retinopathy
        X = df[FEATURE_COLS]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(random_state=42, probability=True),
            "Naive Bayes": GaussianNB(),
        }
        rows = []
        for name, m in models.items():
            m.fit(X_train_s, y_train)
            pred = m.predict(X_test_s)
            prob = m.predict_proba(X_test_s)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X_test_s)
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "F1": f1_score(y_test, pred),
                "AUC": roc_auc_score(y_test, prob),
            })
        results = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

        lr = models["Logistic Regression"]
        rf = models["Random Forest"]
        rf_importance = pd.Series(rf.feature_importances_, index=FEATURE_COLS).rename({
            "systolic_bp": "Systolic BP", "diastolic_bp": "Diastolic BP",
            "cholesterol": "Cholesterol", "age": "Age", "pulse_pressure": "Pulse Pressure",
        }).sort_values(ascending=False)

        df["prognosis_label"] = le.inverse_transform(y)
        group_means = {
            "healthy": df[df.prognosis_label == "no_retinopathy"][FEATURE_COLS[:4]].mean().to_dict(),
            "risk": df[df.prognosis_label == "retinopathy"][FEATURE_COLS[:4]].mean().to_dict(),
        }

        sample = pd.concat([
            df[df.prognosis_label == "no_retinopathy"].sample(min(220, (df.prognosis_label == "no_retinopathy").sum()), random_state=42),
            df[df.prognosis_label == "retinopathy"].sample(min(220, (df.prognosis_label == "retinopathy").sum()), random_state=42),
        ])[FEATURE_COLS[:4] + ["prognosis_label"]]

        return {
            "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
            "lr_coef": lr.coef_[0], "lr_intercept": lr.intercept_[0],
            "results": results, "rf_importance": rf_importance,
            "group_means": group_means, "sample": sample, "live": True,
        }

    # fallback: no dataset.csv found alongside the script
    return {
        "scaler_mean": FALLBACK_SCALER_MEAN, "scaler_scale": FALLBACK_SCALER_SCALE,
        "lr_coef": FALLBACK_LR_COEF, "lr_intercept": FALLBACK_LR_INTERCEPT,
        "results": FALLBACK_RESULTS, "rf_importance": FALLBACK_RF_IMPORTANCE,
        "group_means": FALLBACK_GROUP_MEANS, "sample": None, "live": False,
    }


MODEL = load_model()


def load_dataset():
    path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    if not os.path.exists(path):
        path = "dataset.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


DATASET = load_dataset()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compute_risk(age, sbp, dbp, chol):
    pp = sbp - dbp
    raw = np.array([age, sbp, dbp, chol, pp])
    scaled = (raw - MODEL["scaler_mean"]) / MODEL["scaler_scale"]
    contributions = scaled * MODEL["lr_coef"]
    z = MODEL["lr_intercept"] + contributions.sum()
    prob = sigmoid(z)
    return prob, contributions, pp


def build_live_profile_summary(age, sbp, dbp, chol, contributions):
    healthy = MODEL["group_means"]["healthy"]
    risk = MODEL["group_means"]["risk"]
    current_vals = {
        "Age": age,
        "Systolic BP": sbp,
        "Diastolic BP": dbp,
        "Cholesterol": chol,
    }
    metric_rows = []
    for label, key in [("Age", "age"), ("Systolic BP", "systolic_bp"), ("Diastolic BP", "diastolic_bp"), ("Cholesterol", "cholesterol")]:
        current = current_vals[label]
        healthy_mean = healthy[key]
        risk_mean = risk[key]
        delta = current - healthy_mean
        if abs(current - healthy_mean) <= abs(current - risk_mean):
            risk_label = "low health risk"
        elif abs(current - risk_mean) < abs(current - healthy_mean):
            risk_label = "high health risk"
        else:
            risk_label = "moderate health risk"
        metric_rows.append((label, current, healthy_mean, risk_mean, delta, risk_label))

    top_idx = int(np.argmax(np.abs(np.array(contributions, dtype=float))))
    top_label = FEATURE_LABELS[top_idx]
    top_contrib = contributions[top_idx]
    top_text = "raises the score" if top_contrib >= 0 else "reduces the score"
    return metric_rows, top_label, top_text


def risk_band(prob):
    if prob < 0.35:
        return "LOW", "#2FA898"
    if prob < 0.65:
        return "BORDERLINE", "#E8B454"
    return "ELEVATED", "#D9502F"


def norm_pos(val, rng):
    lo, hi = rng
    return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100.0))


# 2. THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
.stApp {
    background: radial-gradient(circle at 20% -10%, #10201A 0%, #070E0C 55%);
    color: #EDEAE1;
}
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.eyebrow {
    font-size: 11px; letter-spacing: 3px; color: #2FA898; font-weight: 700;
    margin-bottom: 6px; text-transform: uppercase;
}
.panel {
    background: linear-gradient(160deg, #101C18, #0B1512);
    border: 1px solid rgba(232,180,84,0.10);
    border-radius: 16px; padding: 20px 22px; margin-bottom: 18px;
}
.badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    padding: 6px 12px; border-radius: 999px; border: 1px solid rgba(232,180,84,0.25);
    color: #E8B454; margin-right: 8px;
}
.metric-mono { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# 3. HERO
top_l, top_r = st.columns([2.2, 1])
with top_l:
    st.markdown('<div class="eyebrow">Ophthalmic screening model · logistic regression</div>', unsafe_allow_html=True)
    st.markdown("# Diabetic Retinopathy Prediction in Patients")
    st.markdown(
        "This dashboard presents a simple screening workflow for diabetic retinopathy risk, using age, blood pressure, and cholesterol to estimate potential risk and compare the patient against a population reference."
    )
    st.markdown("### Home Page")
    st.markdown("Use the patient details section below to enter the available measurements and review an instant risk assessment, supporting analysis, and model performance summary.")
with top_r:
    best_auc = MODEL["results"]["AUC"].max()
    lr_row = MODEL["results"][MODEL["results"].Model == "Logistic Regression"]
    lr_acc = float(lr_row["Accuracy"].iloc[0]) if not lr_row.empty else 0.7675
    tag = "live-trained" if MODEL["live"] else "pretrained fallback"
    st.markdown(
        f'<div style="text-align:right; margin-top:28px;">'
        f'<span class="badge">AUC {best_auc:.3f}</span>'
        f'<span class="badge">Accuracy {lr_acc*100:.1f}%</span>'
        f'<br><span style="font-size:11px;color:#5C7A70;">model: {tag}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# 4. INPUTS + LIVE GAUGE
col_in, col_gauge = st.columns([1, 1.4])

with col_in:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Patient Details</div>', unsafe_allow_html=True)
    st.markdown("Enter the available clinical values below to generate a screening estimate.")
    patient_id = st.text_input("Patient ID", placeholder="Enter patient ID")

    matched_row = None
    if DATASET is not None and patient_id and str(patient_id).strip():
        patient_id_str = str(patient_id).strip().lower()
        id_matches = DATASET[DATASET["ID"].astype(str).str.strip().str.lower() == patient_id_str]
        if not id_matches.empty:
            matched_row = id_matches.iloc[0]

    default_age = float(matched_row["age"]) if matched_row is not None and "age" in matched_row.index else 60.5
    default_sbp = float(matched_row["systolic_bp"]) if matched_row is not None and "systolic_bp" in matched_row.index else 101.0
    default_dbp = float(matched_row["diastolic_bp"]) if matched_row is not None and "diastolic_bp" in matched_row.index else 90.5
    default_chol = float(matched_row["cholesterol"]) if matched_row is not None and "cholesterol" in matched_row.index else 101.0

    age = st.slider("Age (years)", float(RANGES["age"][0]), float(RANGES["age"][1]), default_age, 0.5)
    sbp = st.slider("Systolic BP (mmHg)", float(RANGES["systolic_bp"][0]), float(RANGES["systolic_bp"][1]), default_sbp, 0.5)
    dbp = st.slider("Diastolic BP (mmHg)", float(RANGES["diastolic_bp"][0]), float(RANGES["diastolic_bp"][1]), default_dbp, 0.5)
    chol = st.slider("Cholesterol (mg/dL)", float(RANGES["cholesterol"][0]), float(RANGES["cholesterol"][1]), default_chol, 0.5)
    prob, contributions, pulse_pressure = compute_risk(age, sbp, dbp, chol)
    prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if matched_row is not None:
        actual_outcome = str(matched_row.get("prognosis", "Unknown")).replace("_", " ")
        st.markdown(
            f'<div style="margin-top:10px;padding:10px;border-radius:10px;background:rgba(47,168,152,0.10);border:1px solid rgba(47,168,152,0.25);font-size:12px;color:#EDEAE1;">'
            f'<strong>Matched dataset record</strong><br/>'
            f'Age: <span class="metric-mono">{matched_row["age"]:.1f}</span> · '
            f'Systolic BP: <span class="metric-mono">{matched_row["systolic_bp"]:.1f}</span> · '
            f'Diastolic BP: <span class="metric-mono">{matched_row["diastolic_bp"]:.1f}</span><br/>'
            f'Cholesterol: <span class="metric-mono">{matched_row["cholesterol"]:.1f}</span> · '
            f'Real outcome: <span class="metric-mono">{actual_outcome}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="margin-top:10px;font-size:12px;color:#5C7A70;">'
        f'Derived pulse pressure: <span class="metric-mono" style="color:#EDEAE1;">{pulse_pressure:.1f} mmHg</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_gauge:
    label, color = risk_band(prob)
    prediction_label = "Retinopathy" if prob >= 0.5 else "No Retinopathy"
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Prediction</div>', unsafe_allow_html=True)
    st.markdown("This estimate updates automatically as you adjust the clinical inputs.")
    st.metric("Risk %", f"{prob * 100:.1f}%")
    st.metric("Prediction", prediction_label)
    st.markdown(
        f'<div style="margin-top:12px;padding:12px;border-radius:12px;background:rgba(232,180,84,0.10);border:1px solid rgba(232,180,84,0.25);">'
        f'<div style="font-size:12px;color:#9CA8A3;">Patient ID: <span class="metric-mono" style="color:#EDEAE1;">{patient_id or "Not provided"}</span></div>'
        f'<div style="font-size:12px;color:#9CA8A3;margin-top:6px;">Prediction: <span class="metric-mono" style="color:#EDEAE1;">{prediction_label}</span></div>'
        f'<div style="font-size:12px;color:#9CA8A3;margin-top:6px;">Risk Score: <span class="metric-mono" style="color:#EDEAE1;">{prob * 100:.1f}%</span></div>'
        f'<div style="font-size:12px;color:#9CA8A3;margin-top:6px;">Prediction Time: <span class="metric-mono" style="color:#EDEAE1;">{prediction_time}</span></div>'
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
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(47,168,152,0.15)"},
                {"range": [35, 65], "color": "rgba(232,180,84,0.15)"},
                {"range": [65, 100], "color": "rgba(217,80,47,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.9, "value": prob * 100},
        },
    ))
    fig.update_layout(
        height=290, margin=dict(t=10, b=0, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#EDEAE1"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="text-align:center; font-size:13px; letter-spacing:2px; font-weight:700; color:{color};">{label} RISK</div>',
        unsafe_allow_html=True,
    )

    # feature contribution strip (vessel-style)
    contrib_html = ""
    for lab, c in zip(FEATURE_LABELS, contributions):
        pos = c >= 0
        dot_color = "#D9502F" if pos else "#2FA898"
        contrib_html += (
            f'<span style="margin-right:16px;font-size:12px;color:#9CA8A3;">'
            f'<span style="color:{dot_color};">●</span> {lab} '
            f'<span class="metric-mono" style="color:#EDEAE1;">{c:+.2f}</span></span>'
        )
    st.markdown(f'<div style="margin-top:8px;text-align:center;">{contrib_html}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 5. RADAR + POPULATION SCATTER
col_radar, col_scatter = st.columns(2)

with col_radar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Patient Comparison</div>', unsafe_allow_html=True)
    st.markdown("#### Patient vs. population norms")

    metrics = ["age", "systolic_bp", "diastolic_bp", "cholesterol"]
    metric_labels = ["Age", "Systolic BP", "Diastolic BP", "Cholesterol"]
    patient_vals = {"age": age, "systolic_bp": sbp, "diastolic_bp": dbp, "cholesterol": chol}

    patient_r = [norm_pos(patient_vals[m], RANGES[m]) for m in metrics] 
    risk_r = [norm_pos(MODEL["group_means"]["risk"][m], RANGES[m]) for m in metrics]
    healthy_r = [norm_pos(MODEL["group_means"]["healthy"][m], RANGES[m]) for m in metrics]

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(r=risk_r + risk_r[:1], theta=metric_labels + metric_labels[:1],
                                     fill="toself", name="Elevated-risk cohort",
                                     line=dict(color="#D9502F"), fillcolor="rgba(217,80,47,0.12)"))
    radar.add_trace(go.Scatterpolar(r=healthy_r + healthy_r[:1], theta=metric_labels + metric_labels[:1],
                                     fill="toself", name="Healthy cohort",
                                     line=dict(color="#2FA898"), fillcolor="rgba(47,168,152,0.12)"))
    radar.add_trace(go.Scatterpolar(r=patient_r + patient_r[:1], theta=metric_labels + metric_labels[:1],
                                     fill="toself", name="Patient",
                                     line=dict(color="#E8B454", width=3), fillcolor="rgba(232,180,84,0.25)"))
    radar.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(237,234,225,0.12)"),
                   angularaxis=dict(gridcolor="rgba(237,234,225,0.12)", color="#9CA8A3")),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(color="#9CA8A3", size=10)),
        paper_bgcolor="rgba(0,0,0,0)", height=340, margin=dict(t=10, b=10, l=40, r=40),
    )
    st.plotly_chart(radar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_scatter:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    label_tag = "live sample" if MODEL["live"] else "440 sampled records"
    st.markdown(f'<div class="eyebrow">Patient Comparison</div>', unsafe_allow_html=True)
    st.markdown("#### Age vs. systolic BP (bubble = cholesterol)")

    scatter = go.Figure()
    if MODEL["sample"] is not None:
        sample = MODEL["sample"]
        for label_name, color_, group in [
            ("No retinopathy", "#2FA898", "no_retinopathy"),
            ("Retinopathy", "#D9502F", "retinopathy"),
        ]:
            g = sample[sample.prognosis_label == group]
            scatter.add_trace(go.Scatter(
                x=g["age"], y=g["systolic_bp"], mode="markers", name=label_name,
                marker=dict(size=g["cholesterol"] / 6, color=color_, opacity=0.5, line=dict(width=0)),
            ))
    scatter.add_trace(go.Scatter(
        x=[age], y=[sbp], mode="markers", name="Patient",
        marker=dict(size=18, color="#E8B454", symbol="star", line=dict(width=1.5, color="#070E0C")),
    ))
    scatter.update_layout(
        xaxis=dict(title="Age", range=[30, 95], gridcolor="rgba(237,234,225,0.08)", color="#9CA8A3"),
        yaxis=dict(title="Systolic BP", range=[65, 155], gridcolor="rgba(237,234,225,0.08)", color="#9CA8A3"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=340,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(color="#9CA8A3", size=10)),
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(scatter, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# 6. MODEL COMPARISON + FEATURE IMPORTANCE
col_models, col_features = st.columns([1.3, 1])

with col_models:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("#### Comparison of all trained models")

    results_sorted = MODEL["results"].sort_values("AUC", ascending=False)
    colors = ["#E8B454" if m == "Logistic Regression" else "rgba(47,168,152,0.55)" for m in results_sorted["Model"]]
    bar = go.Figure(go.Bar(
        x=results_sorted["Model"], y=results_sorted["AUC"], marker_color=colors,
        text=[f"{v:.3f}" for v in results_sorted["AUC"]], textposition="outside",
        textfont=dict(color="#EDEAE1"),
    ))
    bar.update_layout(
        yaxis=dict(range=[0.6, 0.9], title="AUC", gridcolor="rgba(237,234,225,0.06)", color="#9CA8A3"),
        xaxis=dict(color="#9CA8A3", tickangle=-15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300,
        margin=dict(t=20, b=10, l=10, r=10),
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption("Gold bar = Logistic Regression, the model deployed above (best interpretability-to-performance tradeoff).")
    st.markdown("</div>", unsafe_allow_html=True)

with col_features:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown("#### Feature contributions and clinical context")

    metric_rows, top_label, top_text = build_live_profile_summary(age, sbp, dbp, chol, contributions)
    for label, current, healthy_mean, risk_mean, delta, risk_label in metric_rows:
        st.markdown(
            f'<div style="margin-bottom:10px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:12px;color:#9CA8A3;">'
            f'<span>{label}</span><span class="metric-mono" style="color:#EDEAE1;">{current:.1f}</span></div>'
            f'<div style="font-size:11px;color:#5C7A70;margin-top:2px;">'
            f'Healthy mean {healthy_mean:.1f} · Elevated-risk mean {risk_mean:.1f} · {risk_label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="margin-top:12px;padding:10px 12px;border-radius:12px;background:rgba(47,168,152,0.10);border:1px solid rgba(47,168,152,0.2);font-size:12px;color:#EDEAE1;">'
        f'<span style="color:#2FA898;font-weight:700;">Most influential today:</span> {top_label} is the strongest contributor and {top_text}.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

