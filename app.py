import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

# Title and description
st.title("Machine Learning Application Created by Maria Nadeem")
st.write("This application allows you to perform machine learning tasks such as data loading, preprocessing, model selection, training, evaluation, and prediction.")

# Greetings button
if st.button("Greetings"):
    st.balloons()
    st.write("Hello there! Welcome to the Machine Learning Application.")

# Data loading
st.sidebar.header("Data Loading")

# Load data if file is uploaded
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx", "tsv"])
if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        data = pd.read_csv(uploaded_file)
    elif file_extension == "xlsx":
        data = pd.read_excel(uploaded_file)
    elif file_extension == "tsv":
        data = pd.read_csv(uploaded_file, sep="\t")
else:
    default_dataset = st.sidebar.selectbox("Select Default Dataset:", ("Titanic", "Tips", "Iris"))
    if default_dataset == "Titanic":
        data = sns.load_dataset("titanic")
    elif default_dataset == "Tips":
        data = sns.load_dataset("tips")
    else:
        data = sns.load_dataset("iris")

# Basic information about the dataset
st.write("Columns:", data.columns)
st.write("Shape:", data.shape)
st.write("Number of Rows:", data.shape[0])
st.write("Number of columns:", data.shape[1])
st.write("Data Types:", data.dtypes)
st.write("Summary Statistics:", data.describe())
missing_values_percentage = data.isnull().sum() / len(data) * 100
columns_with_missing_values = missing_values_percentage[missing_values_percentage > 0].index.tolist()
st.write("Columns with Missing Values:", columns_with_missing_values)

# Automatic Exploratory Data Analysis
st.write("Exploratory Data Analysis:")
numerical_columns = data.select_dtypes(include=['int', 'float']).columns
fig, axes = plt.subplots(nrows=1, ncols=len(numerical_columns), figsize=(30, 10))
for i, col in enumerate(numerical_columns):
    sns.histplot(data=data[col], ax=axes[i], kde=True)
    axes[i].set_title(col)
st.pyplot(fig)

fig = px.scatter_matrix(data, dimensions=numerical_columns.tolist())
st.plotly_chart(fig)

# Scale the Numeric columns having the float datatype
numerical_columns = data.select_dtypes(include=['float']).columns
# Let's Apply the Standard Scaler
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Encoding categorical variables
for col in data.columns:
    if data[col].dtype == 'object' or data[col].dtype.name == 'category':
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

# Feature and target selection
selected_features = st.sidebar.multiselect("Select Features:", data.columns)
selected_target = st.sidebar.selectbox("Select Target Variable:", data.columns)

# Determine problem type
problem_type = st.sidebar.radio("Select Problem Type:", ("Classification", "Regression"))

if problem_type == "Classification":
    st.write("Problem Type: Classification")
else:
    st.write("Problem Type: Regression")

# Data preprocessing
if st.checkbox("Data Preprocessing"):
    # Handling missing values
    missing_values_percentage = data.isnull().sum() / len(data) * 100
    columns_to_drop = missing_values_percentage[missing_values_percentage > 80].index
    data.drop(columns=columns_to_drop, inplace=True)
    
    selected_features += columns_with_missing_values  # Include columns with missing values
    
    if len(data.columns) > 0:
        imputer = IterativeImputer(max_iter=50)
        data[selected_features] = imputer.fit_transform(data[selected_features])

# Lets check still the Missing values are present or not
missing_values_percentage = data.isnull().sum() / len(data) * 100
st.write('Missing Values present In Data after Imputation:',data.isnull().sum().sum())

# Train test split
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
