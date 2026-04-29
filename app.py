import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="ML Regression Dashboard", layout="wide")
st.title("🚗 Car MPG Prediction Dashboard")
st.markdown("Compare **Linear Regression, Ridge, and Lasso Regression** using Car MPG dataset")

# -----------------------------
# Load Dataset (Local CSV File)
# -----------------------------
@st.cache_data
def load_data():
    file_path = "car_prediction_data.csv"

    if not os.path.exists(file_path):
        st.error("Dataset file 'car_prediction_data.csv' not found. Please keep it in the same folder as app.py")
        st.stop()

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # Remove missing values
    df = df.dropna()

    return df


df = load_data()

st.subheader("📂 Dataset Preview")
st.write("Using uploaded dataset: car_prediction_data.csv")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("📊 Correlation Heatmap")
numeric_df = df.select_dtypes(include=np.number)

fig = px.imshow(
    numeric_df.corr(),
    text_auto=True,
    color_continuous_scale="RdBu_r"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Feature Selection (Dynamic Column Handling)
# -----------------------------
# Clean column names (remove spaces + lowercase)
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

st.subheader("🧾 Available Columns in Dataset")
st.write(df.columns.tolist())

# Update these based on your dataset columns
possible_features = [
    'cylinders',
    'displacement',
    'horsepower',
    'weight',
    'acceleration',
    'model_year'
]

# Keep only columns that exist in dataset
features = [col for col in possible_features if col in df.columns]

if len(features) == 0:
    st.error("No matching feature columns found in your dataset. Please check column names.")
    st.stop()

# Target column check
if 'mpg' not in df.columns:
    st.error("Target column 'mpg' not found in dataset. Please rename your target column correctly.")
    st.stop()

X = df[features]
y = df['mpg']

st.success(f"Selected Features: {features}")

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙ Model Parameters")
ridge_alpha = st.sidebar.slider("Ridge Alpha", 0.01, 50.0, 1.0)
lasso_alpha = st.sidebar.slider("Lasso Alpha", 0.01, 50.0, 1.0)

# Models
lr = LinearRegression()
ridge = Ridge(alpha=ridge_alpha)
lasso = Lasso(alpha=lasso_alpha)

models = {
    "Linear Regression": lr,
    "Ridge Regression": ridge,
    "Lasso Regression": lasso
}

results = []
predictions = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    predictions[name] = pred

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    results.append({
        "Model": name,
        "R² Score": r2,
        "RMSE": rmse
    })

results_df = pd.DataFrame(results)

# -----------------------------
# Model Comparison
# -----------------------------
st.subheader("📈 Model Comparison")
st.dataframe(results_df)

fig_bar = px.bar(
    results_df,
    x="Model",
    y="R² Score",
    color="Model",
    title="R² Score Comparison"
)
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("🎯 Feature Importance")
selected_model = st.selectbox("Select Model", list(models.keys()))

model = models[selected_model]
importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

fig_imp = px.bar(
    importance,
    x="Feature",
    y="Coefficient",
    color="Coefficient",
    title=f"{selected_model} Feature Importance"
)
st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------
# Actual vs Predicted Plot
# -----------------------------
st.subheader("🔍 Actual vs Predicted")
selected_pred = predictions[selected_model]

fig_actual = go.Figure()
fig_actual.add_trace(go.Scatter(
    x=y_test,
    y=selected_pred,
    mode='markers'
))

fig_actual.add_shape(
    type='line',
    x0=y_test.min(),
    y0=y_test.min(),
    x1=y_test.max(),
    y1=y_test.max(),
    line=dict(color='red')
)

fig_actual.update_layout(
    xaxis_title="Actual MPG",
    yaxis_title="Predicted MPG"
)

st.plotly_chart(fig_actual, use_container_width=True)

# -----------------------------
# Residual Analysis
# -----------------------------
st.subheader("📉 Residual Analysis")
residuals = y_test - selected_pred

fig_res = px.histogram(
    residuals,
    nbins=20,
    title="Residual Distribution"
)
st.plotly_chart(fig_res, use_container_width=True)

# -----------------------------
# Real-Time Prediction
# -----------------------------
st.sidebar.header("🚘 Predict MPG")
input_data = []

for feature in features:
    val = st.sidebar.number_input(
        feature,
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )
    input_data.append(val)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)[0]

st.sidebar.success(f"Predicted MPG: {prediction:.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("### 📌 Project Information")
st.write("**Project Name:** Interactive Machine Learning Dashboard")
st.write("**Algorithms Used:** Linear Regression, Ridge Regression, Lasso Regression")
st.write("**Tech Stack:** Python, Streamlit, Scikit-learn, Plotly, Pandas, NumPy")
st.write("**Dataset:** car_prediction_data.csv")

st.markdown("Built with ❤️ for GitHub Portfolio + LinkedIn Showcase")





