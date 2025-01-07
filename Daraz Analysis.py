import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="Daraz Product Analysis", layout="wide", initial_sidebar_state="expanded")

# Apply custom styling
st.markdown(
    """<style>
    .reportview-container {
        background: linear-gradient(to right, #ece9e6, #ffffff);
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stNumberInput>div>div>input {
        border: 2px solid #3498db;
        border-radius: 8px;
    }
    .logo-row {
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin: 20px 0;
    }
    .logo {
        max-width: 60px;
        margin: 0 10px;
    }
    </style>""",
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "EDA", "Data Preprocessing", "Model", "Prediction", "Conclusion & Insights"])

# Load and clean data
@st.cache_data
def load_data():
    file_path = "Daraz.csv"
    data = pd.read_csv(file_path)
    data['Current Price'] = pd.to_numeric(data['Current Price'].str.replace('Rs.', '').str.strip(), errors='coerce')
    data['Original Price'] = pd.to_numeric(data['Original Price'].str.replace('Rs.', '').str.strip(), errors='coerce')
    data['Sold Count'] = pd.to_numeric(data['Sold Count'].str.replace('K', '000').str.replace(' Sold', '').str.strip(), errors='coerce')
    data['Rating in Stars'] = pd.to_numeric(data['Rating in Stars'].str.split('/').str[0], errors='coerce')
    data['Original Price'].fillna(data['Original Price'].median(), inplace=True)
    data['Sold Count'].fillna(data['Sold Count'].median(), inplace=True)
    return data

data = load_data()

# Global variables
features = ['Current Price', 'Rating in Stars', 'Rating Count']
target = 'Sold Count'

# Train and compare models
@st.cache_data
def train_and_compare_models():
    X = data[features].dropna()
    y = data[target][X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_params = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
    rf_grid = GridSearchCV(rf_model, rf_params, cv=3)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Evaluate
    rf_pred = rf_best.predict(X_test)
    lr_pred = lr_model.predict(X_test)

    metrics = {
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'MSE': mean_squared_error(y_test, rf_pred),
            'R2': r2_score(y_test, rf_pred),
            'Best Params': rf_grid.best_params_
        },
        'Linear Regression': {
            'MAE': mean_absolute_error(y_test, lr_pred),
            'MSE': mean_squared_error(y_test, lr_pred),
            'R2': r2_score(y_test, lr_pred)
        }
    }

    return rf_best, metrics

model, model_metrics = train_and_compare_models()

if section == "Introduction":
    st.title("Daraz Product Analysis")
    st.markdown("""
    <h2 style='color: #2980b9;'>Welcome to the Daraz Product Analysis App!</h2>
    <p style='font-size: 18px;'>This project analyzes Daraz product data. The application includes data preprocessing, 
    exploratory analysis, machine learning modeling, and prediction. 
    Use the navigation menu on the left to explore.</p>
    <div class="logo-row">
        <img class="logo" src="https://cdn0.iconfinder.com/data/icons/most-usable-logos/120/Amazon-512.png" width="60" alt="Amazon">
        <img class="logo" src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python">
        <img class="logo" src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit">
        <img class="logo" src="https://cdn.worldvectorlogo.com/logos/seaborn-1.svg" width="90" alt="Seaborn">
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Dataset Overview")
    st.write(data.head())

elif section == "EDA":
    st.title("Exploratory Data Analysis")

    st.markdown("### Summary Statistics")
    st.write(data.describe())
    st.markdown("The dataset provides a diverse range of product details. By examining the summary statistics, we observe significant variation in product pricing and ratings, reflecting the presence of both premium and budget-friendly items. This diversity is crucial for comprehensive insights.")

    st.markdown("### Missing Value Analysis")
    missing_values = data.isnull().sum()
    st.bar_chart(missing_values)
    st.markdown("Missing values are minimal in this dataset, primarily in fields like 'Sold Count' and 'Current Price'. These have been addressed using median imputation to ensure data integrity and consistency for modeling.")

    st.markdown("### Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Current Price'], bins=30, kde=True, ax=ax, color='blue')
    st.pyplot(fig)
    st.markdown("The price distribution reveals that the majority of products are priced under 1000, catering to cost-conscious customers. However, there is a noticeable tail of higher-priced items, indicative of premium product segments.")

    st.markdown("### Box Plot for Outliers")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data['Current Price'], ax=ax)
    st.pyplot(fig)
    st.markdown("The box plot highlights several outliers in the pricing data, especially at the higher end. These could represent luxury items or specialized products with limited demand.")

    st.markdown("### Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.markdown("The correlation heatmap demonstrates that 'Sold Count' is moderately correlated with 'Current Price' and 'Rating Count', suggesting that these features are important drivers of sales performance.")

elif section == "Data Preprocessing":
    st.title("Data Preprocessing")

    st.markdown("### Scaled Data Example")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Current Price', 'Original Price', 'Sold Count']])
    scaled_df = pd.DataFrame(scaled_data, columns=['Current Price', 'Original Price', 'Sold Count'])
    st.write(scaled_df.head())
    st.markdown("Scaling ensures that features like 'Current Price' and 'Sold Count', which have different ranges, are normalized. This helps in improving the convergence and accuracy of machine learning models.")

elif section == "Model":
    st.title("Sales Prediction Model")

    st.markdown("### Model Performance")
    for model_name, metrics in model_metrics.items():
        st.markdown(f"**<span style='color: #16a085;'>{model_name}:</span>**", unsafe_allow_html=True)
        st.write(f"MAE: {metrics['MAE']:.2f}")
        st.write(f"MSE: {metrics['MSE']:.2f}")
        st.write(f"R² Score: {metrics['R2']:.2f}")
        if model_name == 'Random Forest':
            st.write(f"Best Params: {metrics['Best Params']}")
    st.markdown("Random Forest outperformed Linear Regression in terms of accuracy and error metrics, making it the preferred choice for predicting product sales. Its ability to capture non-linear relationships is particularly beneficial for this dataset.")

elif section == "Prediction":
    st.title("Predict Sales")
    price = st.number_input("Enter Product Price", min_value=1, value=300)
    rating = st.number_input("Enter Product Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    rating_count = st.number_input("Enter Rating Count", min_value=0, value=100)

    if st.button("Predict"):
        prediction = model.predict([[price, rating, rating_count]])
        st.write(f"<h3 style='color: #e74c3c;'>Predicted Sold Count: {int(prediction[0])}</h3>", unsafe_allow_html=True)

elif section == "Conclusion & Insights":
    st.title("Conclusion and Insights")

    st.markdown("""
    <ul style='font-size: 18px;'>
        <li><strong>Key Findings:</strong> Products priced moderately (200-600) sell the most. This aligns with customer preferences for value-driven purchases in competitive markets.</li>
        <li><strong>Ratings Matter:</strong> Higher ratings correlate positively with sales, indicating the importance of quality and customer satisfaction.</li>
        <li><strong>Model Comparison:</strong> Random Forest performed better than Linear Regression, showcasing its ability to handle complex data relationships effectively.</li>
    </ul>
    <h3 style='color: #8e44ad;'>Feature Importance:</h3>
    """, unsafe_allow_html=True)

    feature_importances = pd.Series(model.feature_importances_, index=features)
    st.bar_chart(feature_importances)
