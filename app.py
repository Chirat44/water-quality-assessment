# Improved Water Quality Assessment App (Full Replace Code)

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Water Quality Assessment",
    page_icon="💧",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #38bdf8;
}

.sub-title {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.result-safe {
    background-color: #166534;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
    border: 3px solid #22c55e;
}

.result-danger {
    background-color: #991b1b;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 24px;
    font-weight: bold;
    border: 3px solid #ef4444;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.markdown('<div class="main-title">💧 Water Quality Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI System for Checking Drinking Water Quality</div>', unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data

def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

try:
    df = load_data()
except:
    st.error("water_potability.csv file not found")
    st.stop()

# =====================================================
# FEATURES
# =====================================================

FEATURES = [
    'ph',
    'Hardness',
    'Solids',
    'Chloramines',
    'Sulfate',
    'Conductivity',
    'Organic_carbon',
    'Trihalomethanes',
    'Turbidity'
]

TARGET = 'Potability'

# =====================================================
# PREPROCESSING
# =====================================================

X = df[FEATURES]
y = df[TARGET]

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# MODEL METRICS
# =====================================================

pred_test = model.predict(X_test)

accuracy = accuracy_score(y_test, pred_test)
precision = precision_score(y_test, pred_test)
recall = recall_score(y_test, pred_test)
f1 = f1_score(y_test, pred_test)

# =====================================================
# SIDEBAR INPUT
# =====================================================

st.sidebar.header("Input Water Parameters")

ph = st.sidebar.number_input("pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.number_input("Hardness", 0.0, 500.0, 195.0)
solids = st.sidebar.number_input("Solids", 0.0, 50000.0, 22000.0)
chloramines = st.sidebar.number_input("Chloramines", 0.0, 15.0, 7.0)
sulfate = st.sidebar.number_input("Sulfate", 0.0, 500.0, 333.0)
conductivity = st.sidebar.number_input("Conductivity", 0.0, 1000.0, 426.0)
organic_carbon = st.sidebar.number_input("Organic Carbon", 0.0, 30.0, 14.0)
trihalomethanes = st.sidebar.number_input("Trihalomethanes", 0.0, 150.0, 66.0)
turbidity = st.sidebar.number_input("Turbidity", 0.0, 10.0, 4.0)

input_data = {
    'pH': ph,
    'Hardness': hardness,
    'Solids': solids,
    'Chloramines': chloramines,
    'Sulfate': sulfate,
    'Conductivity': conductivity,
    'Organic Carbon': organic_carbon,
    'Trihalomethanes': trihalomethanes,
    'Turbidity': turbidity
}

# =====================================================
# PREDICTION
# =====================================================

input_array = np.array([[
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

input_array = scaler.transform(input_array)

prediction = model.predict(input_array)[0]
probability = model.predict_proba(input_array)[0]
confidence = max(probability) * 100

# =====================================================
# RESULT DISPLAY
# =====================================================

if prediction == 1:
    st.markdown(f'''
    <div class="result-safe">
        ✅ SAFE TO DRINK<br><br>
        Confidence Level: {confidence:.2f}%
    </div>
    ''', unsafe_allow_html=True)
else:
    st.markdown(f'''
    <div class="result-danger">
        ⚠️ NOT SAFE TO DRINK<br><br>
        Confidence Level: {confidence:.2f}%
    </div>
    ''', unsafe_allow_html=True)

# =====================================================
# WATER QUALITY TABLE
# =====================================================

st.markdown("---")
st.subheader("📋 Water Quality Status")

normal_ranges = {
    "pH": (6.5, 8.5),
    "Hardness": (50, 500),
    "Solids": (500, 50000),
    "Chloramines": (0, 10),
    "Sulfate": (200, 400),
    "Conductivity": (200, 800),
    "Organic Carbon": (5, 20),
    "Trihalomethanes": (0, 100),
    "Turbidity": (0, 5)
}

status_data = []

for parameter, value in input_data.items():

    minimum, maximum = normal_ranges[parameter]

    if minimum <= value <= maximum:
        status = "✅ Normal"
        color = "green"
    else:
        status = "⚠️ Abnormal"
        color = "red"

    status_data.append({
        "Parameter": parameter,
        "Value": value,
        "Normal Range": f"{minimum} - {maximum}",
        "Status": status,
        "Color": color
    })

status_df = pd.DataFrame(status_data)

st.dataframe(
    status_df[["Parameter", "Value", "Normal Range", "Status"]],
    use_container_width=True,
    hide_index=True
)

# =====================================================
# GRAPH
# =====================================================

st.markdown("### 📈 Water Parameter Overview")

fig, ax = plt.subplots(figsize=(10, 5))

colors = status_df["Color"]

bars = ax.bar(
    status_df["Parameter"],
    status_df["Value"],
    color=colors
)

for bar in bars:
    height = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{height:.1f}',
        ha='center',
        va='bottom'
    )

plt.xticks(rotation=15)
plt.title("Water Parameters")
plt.ylabel("Values")

st.pyplot(fig)

# =====================================================
# CSV UPLOAD SECTION
# =====================================================

st.markdown("---")
st.subheader("📂 Upload CSV File")

uploaded_file = st.file_uploader(
    "Upload water quality CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.dataframe(uploaded_df.head())

    try:

        upload_X = uploaded_df[FEATURES]

        upload_X = imputer.transform(upload_X)
        upload_X = scaler.transform(upload_X)

        predictions = model.predict(upload_X)
        probabilities = model.predict_proba(upload_X)

        uploaded_df['Prediction'] = [
            'SAFE' if p == 1 else 'NOT SAFE'
            for p in predictions
        ]

        uploaded_df['Confidence'] = [
            round(max(prob) * 100, 2)
            for prob in probabilities
        ]

        st.write("### Prediction Results")

        st.dataframe(uploaded_df)

        csv = uploaded_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download Results",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error: {e}")

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.markdown("---")
st.subheader("🤖 Model Performance")

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.caption("Developed using Streamlit and Machine Learning")

```
