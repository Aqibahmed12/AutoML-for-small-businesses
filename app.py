import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import io
import time
import warnings
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import pickle

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AutoML — Small Businesses", layout="wide", page_icon="🚀")

# Header
st.title("AutoML for Small Businesses — by Aqib Ahmed")
st.write(
    "A clean, modern AutoML playground to explore datasets, build simple pipelines, and export trained models."
)

# -------------------------
# Data loading utilities
# -------------------------
@st.cache_data
def load_default_dataset(name: str) -> pd.DataFrame:
    if name == "Titanic":
        return sns.load_dataset("titanic")
    if name == "Tips":
        return sns.load_dataset("tips")
    return sns.load_dataset("iris")


st.sidebar.header("1) Load data")
uploaded_file = st.sidebar.file_uploader("Upload dataset (csv, xlsx, tsv)", type=["csv", "xlsx", "tsv"])

if uploaded_file is not None:
    try:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "csv":
            data = pd.read_csv(uploaded_file)
        elif ext in ("xls", "xlsx"):
            data = pd.read_excel(uploaded_file)
        elif ext == "tsv":
            data = pd.read_csv(uploaded_file, sep="\t")
        else:
            st.sidebar.error("Unsupported file type; loading Iris dataset instead.")
            data = load_default_dataset("Iris")
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
        data = load_default_dataset("Iris")
else:
    default_dataset = st.sidebar.selectbox("Or pick a default dataset", ("Titanic", "Tips", "Iris"))
    data = load_default_dataset(default_dataset)

# Basic dataset information
st.subheader("Dataset preview")
st.dataframe(data.head())

col1, col2, col3 = st.columns(3)
with col1:
    st.write("Columns:", len(data.columns))
    st.write(list(data.columns)[:10])
with col2:
    st.write("Rows:", data.shape[0])
with col3:
    st.write("Missing cells:", int(data.isnull().sum().sum()))

st.write("Data types:")
st.write(data.dtypes)

missing_pct = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
if missing_pct.max() > 0:
    st.write("Columns with missing values (%):")
    st.table(missing_pct[missing_pct > 0])
else:
    st.write("No missing values detected.")

st.markdown("---")

# -------------------------
# EDA (interactive & safe)
# -------------------------
st.subheader("Exploratory Data Analysis")

numeric_cols = data.select_dtypes(include="number").columns.tolist()
cat_cols = data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

if numeric_cols:
    default_plot_cols = numeric_cols[:3]
    cols_to_plot = st.multiselect("Numeric columns to visualize", numeric_cols, default=default_plot_cols)
    if cols_to_plot:
        fig = px.histogram(data, x=cols_to_plot[0], nbins=30, title=f"Distribution of {cols_to_plot[0]}")
        st.plotly_chart(fig, use_container_width=True)
        if len(cols_to_plot) > 1:
            corr = data[cols_to_plot].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, title="Correlation matrix"), use_container_width=True)
else:
    st.info("No numeric columns available for plotting.")

if len(numeric_cols) >= 2:
    st.write(f"Quick scatter: {numeric_cols[0]} vs {numeric_cols[1]}")
    st.plotly_chart(
        px.scatter(
            data,
            x=numeric_cols[0],
            y=numeric_cols[1],
            color=cat_cols[0] if cat_cols else None,
            title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
        ),
        use_container_width=True,
    )

st.markdown("---")

# -------------------------
# Modeling UI (sidebar)
# -------------------------
st.sidebar.header("2) Modeling & preprocessing")
default_features = (numeric_cols[:2] if numeric_cols else list(data.columns)[:2])
selected_features = st.sidebar.multiselect("Select features (at least 1)", list(data.columns), default=default_features)
selected_target = st.sidebar.selectbox("Select target", list(data.columns), index=min(2, max(0, list(data.columns).index(default_features[0]) if default_features else 0)))
problem_type = st.sidebar.radio("Problem type", ("Regression", "Classification"))

# Preprocess & split
if st.sidebar.button("Preprocess & Split"):
    if not selected_features:
        st.sidebar.error("Choose at least one feature.")
    elif selected_target in selected_features:
        st.sidebar.error("Target cannot be included in features.")
    else:
        X = data[selected_features].copy()
        y = data[selected_target].copy()

        # Drop columns with too many missing values (>80%)
        missing_frac = X.isnull().mean()
        drop_cols = missing_frac[missing_frac > 0.8].index.tolist()
        if drop_cols:
            st.sidebar.warning(f"Dropping columns with >80% missing: {drop_cols}")
            X.drop(columns=drop_cols, inplace=True)

        numeric_features = X.select_dtypes(include="number").columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)],
            remainder="drop",
        )

        test_size = st.sidebar.slider("Test size", 0.05, 0.5, 0.2)
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            # Save to session so we can train later
            st.session_state.update(
                {
                    "X_train_raw": X_train_raw,
                    "X_test_raw": X_test_raw,
                    "y_train": y_train,
                    "y_test": y_test,
                    "preprocessor": preprocessor,
                    "numeric_features": numeric_features,
                    "categorical_features": categorical_features,
                }
            )
            st.sidebar.success("Preprocessing setup saved to session.")
        except Exception as e:
            st.sidebar.error(f"Split failed: {e}")

st.sidebar.markdown("---")

# Model selection
st.sidebar.header("3) Choose model")
if problem_type == "Regression":
    model_name = st.sidebar.selectbox("Regression model", ("Linear Regression", "SVR", "Decision Tree", "Random Forest"))
else:
    model_name = st.sidebar.selectbox("Classification model", ("Logistic Regression", "SVC", "Decision Tree", "Random Forest"))

# Train model
if st.button("Train model"):
    if "X_train_raw" not in st.session_state:
        st.error("Run Preprocess & Split first.")
    else:
        X_train_raw = st.session_state.X_train_raw
        X_test_raw = st.session_state.X_test_raw
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        preprocessor = st.session_state.preprocessor

        # Choose estimator
        if problem_type == "Regression":
            if model_name == "Linear Regression":
                estimator = LinearRegression()
            elif model_name == "SVR":
                estimator = SVR()
            elif model_name == "Decision Tree":
                estimator = DecisionTreeRegressor(random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            if model_name == "Logistic Regression":
                estimator = LogisticRegression(max_iter=500)
            elif model_name == "SVC":
                estimator = SVC(probability=True)
            elif model_name == "Decision Tree":
                estimator = DecisionTreeClassifier(random_state=42)
            else:
                estimator = RandomForestClassifier(n_estimators=100, random_state=42)

        # Full pipeline: preprocessor + estimator
        model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

        try:
            model_pipeline.fit(X_train_raw, y_train)
            st.session_state.trained_pipeline = model_pipeline
            st.success("Model trained and stored in session as 'trained_pipeline'.")

            # Evaluate
            y_pred = model_pipeline.predict(X_test_raw)
            st.subheader("Evaluation")
            if problem_type == "Regression":
                st.write("MSE:", mean_squared_error(y_test, y_pred))
                st.write("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
                st.write("MAE:", mean_absolute_error(y_test, y_pred))
                st.write("R2:", r2_score(y_test, y_pred))
            else:
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
                st.write("Precision (weighted):", precision_score(y_test, y_pred, average="weighted", zero_division=0))
                st.write("Recall (weighted):", recall_score(y_test, y_pred, average="weighted", zero_division=0))
                st.write("F1 (weighted):", f1_score(y_test, y_pred, average="weighted", zero_division=0))
                st.write("Confusion matrix:")
                st.write(confusion_matrix(y_test, y_pred))
        except Exception as e:
            st.error(f"Training failed: {e}")

# Export trained pipeline
st.sidebar.header("4) Export")
if "trained_pipeline" in st.session_state:
    st.sidebar.success("Trained pipeline ready")
    buffer = io.BytesIO()
    pickle.dump(st.session_state.trained_pipeline, buffer)
    buffer.seek(0)
    st.sidebar.download_button("Download trained pipeline (.pkl)", buffer, file_name="trained_pipeline.pkl")
else:
    st.sidebar.info("Train a model to enable export")

st.markdown("---")

# Prediction playground for single records
st.subheader("Predict on a single record (playground)")
if "trained_pipeline" in st.session_state:
    model = st.session_state.trained_pipeline
    sample_input = {}
    # Use the last selected features (keep UX simple)
    if selected_features:
        for col in selected_features:
            if col in numeric_cols:
                sample_input[col] = st.number_input(f"{col}", value=float(data[col].dropna().mean()) if not data[col].dropna().empty else 0.0)
            else:
                uniques = data[col].dropna().unique().tolist()
                if uniques:
                    sample_input[col] = st.selectbox(f"{col}", options=uniques)
                else:
                    sample_input[col] = st.text_input(f"{col} (no unique values available)")
        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([sample_input])
                pred = model.predict(input_df)
                st.success(f"Prediction: {pred.tolist()}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("No features selected for prediction.")
else:
    st.info("Train a model to use the prediction playground.")

st.markdown("---")

# Creative expensive computation demo (cached)
@st.cache_resource
def expensive_computation(seed: int = 42):
    # Simulate a heavier computation
    time.sleep(1.5)
    rng = np.random.RandomState(seed)
    return rng.normal(size=1000)


if st.button("Run creative expensive computation"):
    with st.spinner("Creating something colorful... 🎨"):
        arr = expensive_computation()
        st.success("Done — a playful sample was produced")
        fig = px.histogram(arr, nbins=30, title="Result distribution from creative computation")
        st.plotly_chart(fig, use_container_width=True)

st.caption("Built with ❤️ by Aqib Ahmed — fork and extend.")
test_size = st.sidebar.slider("Select Train Test Split Size:", 0.1, 0.9, 0.2)
X_train, X_test, y_train, y_test = train_test_split(data[selected_features], data[selected_target], test_size=test_size, random_state=42)

# Model selection
if st.checkbox("Model Selection"):
    if problem_type == "Regression":
        model_name = st.sidebar.selectbox("Select Regression Model:", ("Linear Regression", "SVR", "Decision Tree", "Random Forest"))
    else:
        model_name = st.sidebar.selectbox("Select Classification Model:", ("Support Vector Classifier", "Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"))

# Model training and evaluation
if st.button("Train Model"):
    if problem_type == "Regression":
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "SVR":
            model = SVR()
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor()
        else:
            model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation metrics
        st.write("Evaluation Metrics:")
        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        st.write("Root Mean Squared Error:", mean_squared_error(y_test, y_pred, squared=False))
        st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        st.write("R2 Score:", r2_score(y_test, y_pred))
    else:
        if model_name == "Support Vector Classifier":
            model = SVC()
        elif model_name == "Logistic Regression":
            model = LogisticRegression()
        elif model_name == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        else:
            model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation metrics
        st.write("Evaluation Metrics:")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

# Prediction function
def predict(model, input_data):
    # Make predictions
    predictions = model.predict(input_data)
    return predictions

# Save Model Button
if st.button("Save Model"):
    with open('trained_model.pkl', 'wb') as file:
        pickle.dump(model_name, file)
    st.write("Model saved successfully.")
# Cache Button
@st.cache_resource
def expensive_computation():
    # Perform expensive computation here
    pass

if st.button("Run Expensive Computation"):
    expensive_computation()
    st.write("Expensive computation completed.")



