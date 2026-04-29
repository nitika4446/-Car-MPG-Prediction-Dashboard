import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Car Price Prediction Dashboard",
    layout="wide"
)

st.title("🚗 Car Price Prediction Dashboard")
st.markdown("""
Compare **Linear Regression**, **Ridge Regression**, and **Lasso Regression**
using a Car Price dataset with interactive visualizations.
""")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    file_path = "car_prediction_data.csv"

    if not os.path.exists(file_path):
        st.error("Dataset file 'car_prediction_data.csv' not found.")
        st.stop()

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop missing values
    df = df.dropna()

    return df


df = load_data()

# -----------------------------
# Dataset Preview
# -----------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

st.subheader("Available Columns")
st.write(df.columns.tolist())

# -----------------------------
# Target Column
# -----------------------------
TARGET_COLUMN = "selling_price"

if TARGET_COLUMN not in df.columns:
    st.error(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    st.stop()

# Drop unnecessary text column
if "car_name" in df.columns:
    df = df.drop("car_name", axis=1)

# -----------------------------
# Handle Categorical Columns
# -----------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns

if len(categorical_cols) > 0:
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
st.subheader("📊 Correlation Heatmap")

numeric_df = df.select_dtypes(include=np.number)

fig_heatmap = px.imshow(
    numeric_df.corr(),
    text_auto=True,
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Sidebar Parameters
# -----------------------------
st.sidebar.header("⚙ Model Parameters")

ridge_alpha = st.sidebar.slider(
    "Ridge Alpha",
    min_value=0.01,
    max_value=100.0,
    value=1.0
)

lasso_alpha = st.sidebar.slider(
    "Lasso Alpha",
    min_value=0.01,
    max_value=100.0,
    value=1.0
)

# -----------------------------
# Models
# -----------------------------
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
    title="Model Performance Comparison"
)

st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("🎯 Feature Importance")

selected_model = st.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[selected_model]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

fig_importance = px.bar(
    importance_df,
    x="Feature",
    y="Coefficient",
    color="Coefficient",
    title=f"{selected_model} Feature Importance"
)

st.plotly_chart(fig_importance, use_container_width=True)

# -----------------------------
# Actual vs Predicted Plot
# -----------------------------
st.subheader("🔍 Actual vs Predicted")

selected_predictions = predictions[selected_model]

fig_actual = go.Figure()

fig_actual.add_trace(
    go.Scatter(
        x=y_test,
        y=selected_predictions,
        mode="markers",
        name="Predictions"
    )
)

fig_actual.add_shape(
    type="line",
    x0=y_test.min(),
    y0=y_test.min(),
    x1=y_test.max(),
    y1=y_test.max(),
    line=dict(color="red")
)

fig_actual.update_layout(
    xaxis_title="Actual Price",
    yaxis_title="Predicted Price"
)

st.plotly_chart(fig_actual, use_container_width=True)

# -----------------------------
# Residual Analysis
# -----------------------------
st.subheader("📉 Residual Analysis")

residuals = y_test - selected_predictions

fig_residual = px.histogram(
    residuals,
    nbins=30,
    title="Residual Distribution"
)

st.plotly_chart(fig_residual, use_container_width=True)

# -----------------------------
# Real-Time Prediction
# -----------------------------
st.sidebar.header("🚘 Predict Car Price")

input_data = []

for feature in X.columns:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())

    value = st.sidebar.number_input(
        feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

prediction = model.predict(input_scaled)[0]

st.sidebar.success(f"Predicted Car Price: ₹ {prediction:,.2f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("### 📌 Project Information")
st.write("**Project Name:** Car Price Prediction Dashboard")
st.write("**Algorithms Used:** Linear Regression, Ridge Regression, Lasso Regression")
st.write("**Tech Stack:** Python, Streamlit, Scikit-learn, Plotly, Pandas, NumPy")
st.write("**Dataset:** car_prediction_data.csv")

st.markdown("Built with ❤️ for GitHub Portfolio & LinkedIn Showcase")



