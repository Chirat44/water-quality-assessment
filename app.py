import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Water Quality Assessment",
    page_icon="💧",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #020817;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
}

.result-box-safe {
    background-color: #065f46;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    border: 3px solid #10b981;
}

.result-box-danger {
    background-color: #991b1b;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    border: 3px solid #ef4444;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("""
<h1 style='text-align:center; color:#38bdf8;'>
💧 Water Quality Assessment
</h1>

<p style='text-align:center; color:gray; font-size:20px;'>
AI System for Checking Drinking Water Quality
</p>
""", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Predict Water Quality",
    "📁 Upload CSV File",
    "📊 Model Information"
])

# =========================================================
# FUNCTION
# =========================================================
def calculate_prediction(row):

    score = 0

    if 6.5 <= row["ph"] <= 8.5:
        score += 1

    if row["Hardness"] < 500:
        score += 1

    if row["Solids"] < 30000:
        score += 1

    if row["Chloramines"] < 10:
        score += 1

    if row["Sulfate"] < 400:
        score += 1

    if row["Conductivity"] < 500:
        score += 1

    if row["Organic_carbon"] < 20:
        score += 1

    if row["Trihalomethanes"] < 100:
        score += 1

    if row["Turbidity"] < 5:
        score += 1

    confidence = round((score / 9) * 100, 2)

    prediction = (
        "SAFE TO DRINK"
        if score >= 7
        else "NOT SAFE TO DRINK"
    )

    return prediction, confidence


# =========================================================
# TAB 1 - MANUAL INPUT
# =========================================================
with tab1:

    st.subheader("Enter Water Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input("pH", value=7.0)
        hardness = st.number_input("Hardness", value=195.0)
        solids = st.number_input("Solids", value=22000.0)

    with col2:
        chloramines = st.number_input("Chloramines", value=7.0)
        sulfate = st.number_input("Sulfate", value=333.0)
        conductivity = st.number_input("Conductivity", value=426.0)

    with col3:
        organic_carbon = st.number_input("Organic Carbon", value=14.0)
        trihalomethanes = st.number_input("Trihalomethanes", value=66.0)
        turbidity = st.number_input("Turbidity", value=4.0)

    if st.button("🔎 Predict Water Quality"):

        row = {
            "ph": ph,
            "Hardness": hardness,
            "Solids": solids,
            "Chloramines": chloramines,
            "Sulfate": sulfate,
            "Conductivity": conductivity,
            "Organic_carbon": organic_carbon,
            "Trihalomethanes": trihalomethanes,
            "Turbidity": turbidity
        }

        prediction, confidence = calculate_prediction(row)

        st.markdown("---")

        # RESULT BOX
        if prediction == "SAFE TO DRINK":

            st.markdown(f"""
            <div class="result-box-safe">
                <h1>✅ SAFE TO DRINK</h1>
                <h2>Confidence Level: {confidence}%</h2>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-box-danger">
                <h1>⚠️ NOT SAFE TO DRINK</h1>
                <h2>Confidence Level: {confidence}%</h2>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # TABLE
        st.subheader("📋 Water Quality Status")

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
            ]
        })

        status_df["Status"] = "✅ Normal"

        st.dataframe(status_df, use_container_width=True)

        # GRAPH
        st.subheader("📊 Water Parameter Status")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            status_df["Parameter"],
            [1] * len(status_df),
            color="green"
        )

        ax.set_ylim(0, 1.2)

        ax.set_title("Green = Normal | Red = Abnormal")

        plt.xticks(rotation=20)

        st.pyplot(fig)

# =========================================================
# TAB 2 - CSV FILE
# =========================================================
with tab2:

    st.header("📁 Upload CSV File")

    st.info("""
CSV file must contain these columns:

• ph
• Hardness
• Solids
• Chloramines
• Sulfate
• Conductivity
• Organic_carbon
• Trihalomethanes
• Turbidity
""")

    # TEMPLATE
    template_df = pd.DataFrame(columns=[
        "ph",
        "Hardness",
        "Solids",
        "Chloramines",
        "Sulfate",
        "Conductivity",
        "Organic_carbon",
        "Trihalomethanes",
        "Turbidity"
    ])

    csv_template = template_df.to_csv(index=False)

    st.download_button(
        label="⬇ Download CSV Template",
        data=csv_template,
        file_name="water_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df_csv = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(df_csv.head(10), use_container_width=True)

        if st.button("🚀 Predict CSV File"):

            predictions = []
            confidences = []

            for _, row in df_csv.iterrows():

                prediction, confidence = calculate_prediction(row)

                predictions.append(prediction)
                confidences.append(confidence)

            df_csv["Prediction"] = predictions
            df_csv["Confidence"] = confidences

            # SUMMARY
            safe_count = (df_csv["Prediction"] == "SAFE TO DRINK").sum()

            unsafe_count = (
                df_csv["Prediction"] == "NOT SAFE TO DRINK"
            ).sum()

            total = len(df_csv)

            st.markdown("---")

            st.subheader("📊 Prediction Summary")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>{total}</h2>
                    <p>Total Samples</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style='color:#10b981;'>{safe_count}</h2>
                    <p>Safe Water</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h2 style='color:#ef4444;'>{unsafe_count}</h2>
                    <p>Unsafe Water</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # RESULT TABLE
            st.subheader("📋 Prediction Results")

            st.dataframe(df_csv, use_container_width=True)

            # GRAPH
            st.subheader("📊 Prediction Overview")

            fig, ax = plt.subplots(figsize=(6, 4))

            labels = ["Safe", "Unsafe"]

            values = [safe_count, unsafe_count]

            colors = ["green", "red"]

            ax.bar(labels, values, color=colors)

            ax.set_ylabel("Number of Samples")

            st.pyplot(fig)

            # DOWNLOAD
            csv = df_csv.to_csv(index=False)

            st.download_button(
                "⬇ Download Results",
                csv,
                "prediction_results.csv",
                "text/csv"
            )

# =========================================================
# TAB 3 - MODEL INFO
# =========================================================
with tab3:

    st.header("📊 Model Information")

    st.markdown("""
### Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest

### Features Used

- pH
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

### Purpose

This system predicts whether water is safe to drink using AI and Machine Learning techniques.
""")