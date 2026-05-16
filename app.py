import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Water Quality Assessment",
    page_icon="💧",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#38bdf8;
}

.sub-title {
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
    font-size:18px;
}

.result-good {
    background-color:#14532d;
    color:white;
    padding:30px;
    border-radius:18px;
    border:2px solid #22c55e;
    text-align:center;
}

.result-bad {
    background-color:#7f1d1d;
    color:white;
    padding:30px;
    border-radius:18px;
    border:2px solid #ef4444;
    text-align:center;
}

.big-text {
    font-size:40px;
    font-weight:bold;
}

.small-text {
    font-size:22px;
    margin-top:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity"
]

TARGET = "Potability"

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_resource
def load_and_train():

    df = pd.read_csv("water_potability.csv")

    X = df[FEATURES]
    y = df[TARGET]

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1-Score": f1_score(y_test, preds)
    }

    return model, metrics, imputer, scaler


model, metrics, imputer, scaler = load_and_train()

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">💧 Water Quality Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI System for Checking Drinking Water Quality</div>',
    unsafe_allow_html=True
)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Predict Water Quality",
    "📂 Upload CSV File",
    "📊 Model Information"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("Enter Water Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", 0.0, 14.0, 7.0)
        hardness = st.number_input("Hardness", 0.0, 500.0, 195.0)
        solids = st.number_input("Solids", 0.0, 60000.0, 22000.0)

    with col2:
        chloramines = st.number_input("Chloramines", 0.0, 15.0, 7.0)
        sulfate = st.number_input("Sulfate", 0.0, 500.0, 333.0)
        conductivity = st.number_input("Conductivity", 0.0, 900.0, 426.0)

    with col3:
        organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 14.0)
        trihalomethanes = st.number_input("Trihalomethanes", 0.0, 130.0, 66.0)
        turbidity = st.number_input("Turbidity", 0.0, 10.0, 4.0)

    if st.button("🔍 Predict Water Quality", use_container_width=True):

        row = np.array([[
            ph,
            hardness,
            solids,
            chloramines,
            sulfate,
            conductivity,
            organic_carbon,
            trihalomethanes,
            turbidity
        ]])

        row = imputer.transform(row)
        row = scaler.transform(row)

        prediction = model.predict(row)[0]
        probability = model.predict_proba(row)[0]

        confidence = probability[int(prediction)] * 100

        st.divider()

        if prediction == 1:

            st.markdown(f"""
            <div class="result-good">
                <div class="big-text">✅ SAFE TO DRINK</div>
                <div class="small-text">
                    Confidence Level: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-bad">
                <div class="big-text">⚠️ NOT SAFE TO DRINK</div>
                <div class="small-text">
                    Confidence Level: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # =====================================================
        # STATUS TABLE
        # =====================================================

        st.subheader("Water Quality Status")

        status_df = pd.DataFrame({

            "Parameter": [
                "pH",
                "Hardness",
                "Solids",
                "Chloramines",
                "Sulfate",
                "Conductivity",
                "Organic Carbon",
                "Trihalomethanes",
                "Turbidity"
            ],

            "Value": [
                ph,
                hardness,
                solids,
                chloramines,
                sulfate,
                conductivity,
                organic_carbon,
                trihalomethanes,
                turbidity
            ],

            "Status": [
                "Normal" if 6.5 <= ph <= 8.5 else "Abnormal",
                "Normal" if hardness <= 300 else "High",
                "Normal" if solids <= 30000 else "High",
                "Normal" if chloramines <= 10 else "High",
                "Normal" if sulfate <= 400 else "High",
                "Normal" if conductivity <= 500 else "High",
                "Normal" if organic_carbon <= 20 else "High",
                "Normal" if trihalomethanes <= 100 else "High",
                "Normal" if turbidity <= 5 else "High"
            ]
        })

        st.dataframe(
            status_df,
            use_container_width=True
        )

        # =====================================================
        # STATUS GRAPH
        # =====================================================

        st.subheader("Water Parameter Status")

        fig, ax = plt.subplots(figsize=(10,4))

        colors = [
            "#22c55e" if s == "Normal" else "#ef4444"
            for s in status_df["Status"]
        ]

        ax.bar(
            status_df["Parameter"],
            [1] * len(status_df),
            color=colors
        )

        ax.set_yticks([])

        ax.set_title("Green = Normal | Red = Abnormal")

        plt.xticks(rotation=20)

        st.pyplot(fig)

# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.subheader("Upload CSV File")

    st.info("""
CSV file must contain these columns:

- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
""")

    template = pd.DataFrame(columns=FEATURES)

    csv = template.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download CSV Template",
        csv,
        "water_template.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(df)

        if st.button("🚀 Predict CSV File", use_container_width=True):

            X = df[FEATURES]

            X = imputer.transform(X)
            X = scaler.transform(X)

            preds = model.predict(X)

            probs = model.predict_proba(X)

            results = []

            for i in range(len(preds)):

                results.append({

                    "Prediction":
                        "SAFE" if preds[i] == 1 else "NOT SAFE",

                    "Confidence":
                        f"{max(probs[i])*100:.2f}%"

                })

            result_df = pd.concat([
                df,
                pd.DataFrame(results)
            ], axis=1)

            st.subheader("Prediction Results")

            st.dataframe(result_df)

            csv_result = result_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇ Download Results",
                csv_result,
                "prediction_results.csv",
                "text/csv"
            )

# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.subheader("Model Performance")

    metric_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": list(metrics.values())
    })

    st.dataframe(metric_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7,4))

    sns.barplot(
        x="Metric",
        y="Value",
        data=metric_df,
        palette="Blues",
        ax=ax
    )

    ax.set_ylim(0,1)

    st.pyplot(fig)

    st.divider()

    st.subheader("About This Project")

    st.write("""
This project uses Artificial Intelligence and Machine Learning
to determine whether water is safe for drinking.

Model Used:
- Random Forest Classifier

Dataset:
- Kaggle Water Potability Dataset

Developed for Computer Engineering Senior Project.
""")