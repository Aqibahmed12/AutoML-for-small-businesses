import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import pickle
import warnings

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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

warnings.filterwarnings("ignore")

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="NexaBuild AutoML",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9; font-weight: 700;}
    .sub-header {font-size: 1.5rem; color: #333;}
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helper Functions
# -------------------------
@st.cache_data
def load_dataset(file_upload, default_choice):
    if file_upload is not None:
        try:
            ext = file_upload.name.split(".")[-1].lower()
            if ext == "csv":
                return pd.read_csv(file_upload)
            elif ext in ["xls", "xlsx"]:
                return pd.read_excel(file_upload)
            elif ext == "tsv":
                return pd.read_csv(file_upload, sep="\t")
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    
    # Fallback to default datasets
    import seaborn as sns
    if default_choice == "Titanic":
        return sns.load_dataset("titanic")
    elif default_choice == "Tips":
        return sns.load_dataset("tips")
    elif default_choice == "Iris":
        return sns.load_dataset("iris")
    return None

def plot_feature_importance(model, feature_names):
    """Smartly plots feature importance or coefficients depending on model type"""
    importances = None
    
    # Check for tree-based feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    # Check for linear coefficients
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        
    if importances is not None:
        # Handle OneHotEncoder feature expansion vs original names
        # Simple hack: just plot top N features if lengths don't match exactly or just show raw
        # For simplicity in this demo, we verify length match or truncate
        if len(importances) != len(feature_names):
             st.warning("Feature expansion occurred (OneHotEncoding). Showing raw importance indices.")
             feature_names = [f"Feature {i}" for i in range(len(importances))]

        df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        df_imp = df_imp.sort_values(by="Importance", ascending=True)
        
        fig = px.bar(df_imp, x="Importance", y="Feature", orientation='h', title="Feature Importance / Coefficients")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this specific model type.")

# -------------------------
# Sidebar: Setup
# -------------------------
st.sidebar.markdown("## ⚙️ Configuration")

# 1. Data Loading
st.sidebar.subheader("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Data", type=["csv", "xlsx"])
default_ds = st.sidebar.selectbox("Or use sample data", ["Iris", "Titanic", "Tips"])

# Load Data
df = load_dataset(uploaded_file, default_ds)

if df is not None:
    # 2. Problem Definition
    st.sidebar.subheader("2. Problem Definition")
    all_cols = df.columns.tolist()
    target_col = st.sidebar.selectbox("Target Column (Prediction)", all_cols, index=len(all_cols)-1)
    
    # Guess problem type
    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
    default_prob_type = "Regression" if is_numeric_target and df[target_col].nunique() > 10 else "Classification"
    
    problem_type = st.sidebar.radio("Problem Type", ["Regression", "Classification"], index=0 if default_prob_type == "Regression" else 1)
    
    # Drop columns
    drop_cols = st.sidebar.multiselect("Drop Columns (ID, etc.)", [c for c in all_cols if c != target_col])
    
    # 3. Model Config
    st.sidebar.subheader("3. Model Selection")
    if problem_type == "Regression":
        model_name = st.sidebar.selectbox("Algorithm", ["Linear Regression", "Random Forest", "Decision Tree", "SVR"])
    else:
        model_name = st.sidebar.selectbox("Algorithm", ["Logistic Regression", "Random Forest", "Decision Tree", "SVC"])
        
    split_size = st.sidebar.slider("Test Set Size", 0.1, 0.5, 0.2)

else:
    st.stop()

# -------------------------
# Main Interface
# -------------------------
st.markdown("<div class='main-header'>🚀 NexaBuild AutoML</div>", unsafe_allow_html=True)
st.write(f"Active Dataset: **{uploaded_file.name if uploaded_file else default_ds}** | Target: **{target_col}**")

# Tabs for better organization
tab1, tab2, tab3 = st.tabs(["📊 Explore Data", "🧠 Train Model", "🔮 Predict"])

# ========================
# TAB 1: EDA
# ========================
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())
    
    with st.expander("👀 View Raw Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribution Analysis")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            selected_num = st.selectbox("Select Numeric Column", numeric_cols)
            fig_hist = px.histogram(df, x=selected_num, color=target_col if df[target_col].nunique() < 10 else None, marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with c2:
        st.subheader("Correlation Heatmap")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation.")

# ========================
# TAB 2: TRAINING
# ========================
with tab2:
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Preprocessing and Training..."):
            # 1. Data Prep
            X = df.drop(columns=[target_col] + drop_cols)
            y = df[target_col]
            
            # Identify types
            num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Preprocessing Pipelines
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_transformer, num_features),
                    ('cat', cat_transformer, cat_features)
                ]
            )
            
            # Label Encode Target if Classification and Text
            le = None
            if problem_type == "Classification" and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=42)
            
            # Model Selection
            if problem_type == "Regression":
                if model_name == "Linear Regression": model = LinearRegression()
                elif model_name == "Random Forest": model = RandomForestRegressor(n_estimators=100)
                elif model_name == "Decision Tree": model = DecisionTreeRegressor()
                elif model_name == "SVR": model = SVR()
            else:
                if model_name == "Logistic Regression": model = LogisticRegression(max_iter=1000)
                elif model_name == "Random Forest": model = RandomForestClassifier(n_estimators=100)
                elif model_name == "Decision Tree": model = DecisionTreeClassifier()
                elif model_name == "SVC": model = SVC(probability=True)

            # Create Main Pipeline
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            # Train
            clf.fit(X_train, y_train)
            
            # Store in Session State
            st.session_state['model'] = clf
            st.session_state['features'] = X.columns.tolist()
            st.session_state['num_features'] = num_features
            st.session_state['cat_features'] = cat_features
            st.session_state['target_name'] = target_col
            st.session_state['problem_type'] = problem_type
            st.session_state['label_encoder'] = le  # Save label encoder for decoding predictions later
            
            # Predict
            y_pred = clf.predict(X_test)
            
            st.success("Training Complete!")
            st.markdown("---")
            
            # Metrics
            col_m1, col_m2, col_m3 = st.columns(3)
            
            if problem_type == "Regression":
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                col_m1.metric("RMSE", f"{rmse:.4f}")
                col_m2.metric("MAE", f"{mae:.4f}")
                col_m3.metric("R² Score", f"{r2:.4f}")
                
                # Actual vs Predicted Plot
                fig_res = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                fig_res.add_shape(type="line", line=dict(dash="dash"), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
                st.plotly_chart(fig_res, use_container_width=True)
                
            else:
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                col_m1.metric("Accuracy", f"{acc:.2%}")
                col_m2.metric("Precision", f"{prec:.2%}")
                col_m3.metric("F1 Score", f"{f1:.2%}")
                
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix", color_continuous_scale='Blues')
                st.plotly_chart(fig_cm, use_container_width=True)

            # Feature Importance
            st.markdown("### 🔍 Model Insights")
            try:
                # Extract feature names after preprocessing
                # This is tricky with pipelines, so we do a best-effort approach
                # accessing the steps...
                
                # Plot
                final_model = clf.named_steps['classifier']
                
                # Getting transformed feature names is complex, passing raw numerical + categorical names
                # Ideally, we get feature names from OneHotEncoder
                ohe = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
                cat_names = ohe.get_feature_names_out(cat_features)
                final_feature_names = num_features + list(cat_names)
                
                plot_feature_importance(final_model, final_feature_names)
            except Exception as e:
                st.info("Could not extract detailed feature importance for this specific pipeline configuration.")

# ========================
# TAB 3: PREDICTION
# ========================
with tab3:
    if 'model' in st.session_state:
        st.subheader("Make New Predictions")
        
        pred_type = st.radio("Input Method", ["Manual Entry", "Batch Upload (CSV)"], horizontal=True)
        
        if pred_type == "Manual Entry":
            with st.form("prediction_form"):
                input_data = {}
                cols = st.columns(3)
                for i, col in enumerate(st.session_state['features']):
                    with cols[i % 3]:
                        # Check original type in dataframe to render correct widget
                        if col in st.session_state['num_features']:
                            input_data[col] = st.number_input(f"{col}", value=0.0)
                        else:
                            # Try to find unique values from original DF if possible, else text
                            unique_vals = df[col].dropna().unique().tolist()
                            input_data[col] = st.selectbox(f"{col}", unique_vals)
                
                submit = st.form_submit_button("Predict")
                
                if submit:
                    input_df = pd.DataFrame([input_data])
                    prediction = st.session_state['model'].predict(input_df)
                    
                    # Decode if classification
                    if st.session_state['label_encoder'] is not None:
                         final_val = st.session_state['label_encoder'].inverse_transform(prediction)[0]
                    else:
                         final_val = prediction[0]
                         
                    st.success(f"Predicted {st.session_state['target_name']}: **{final_val}**")
        
        else: # Batch Upload
            batch_file = st.file_uploader("Upload CSV for scoring (must have same columns as training data)", type="csv")
            if batch_file:
                batch_df = pd.read_csv(batch_file)
                if st.button("Score File"):
                    try:
                        preds = st.session_state['model'].predict(batch_df)
                        if st.session_state['label_encoder'] is not None:
                            preds = st.session_state['label_encoder'].inverse_transform(preds)
                            
                        batch_df['Prediction'] = preds
                        st.dataframe(batch_df.head())
                        
                        # Download
                        csv = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Error during prediction: {e}. Check if columns match.")
            
    else:
        st.info("👈 Please train a model in the 'Train Model' tab first.")

# Footer
st.markdown("---")
st.caption("AutoML Framework • Built with Streamlit & Scikit-Learn")
