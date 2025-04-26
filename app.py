import streamlit as st
import pandas as pd
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import os
import numpy as np
import pickle
from pycaret.classification import models

# Initialize session state
if 'user_credentials' not in st.session_state:
    st.session_state['user_credentials'] = {}

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "login"

if 'df' not in st.session_state:
    st.session_state['df'] = None

if 'file_name' not in st.session_state:
    st.session_state['file_name'] = None

if 'removed_features' not in st.session_state:
    st.session_state['removed_features'] = []

if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None

# Load user credentials from file
def load_credentials():
    if os.path.exists('credentials.pkl'):
        with open('credentials.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return {}

# Save user credentials to file
def save_credentials(credentials):
    with open('credentials.pkl', 'wb') as f:
        pickle.dump(credentials, f)

# Function to switch pages
def switch_page(page):
    st.session_state['current_page'] = page

# Sign-In page
def sign_in():
    st.title("Sign-In")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Sign-In"):
        if new_username and new_password:
            credentials = load_credentials()
            if new_username in credentials:
                st.warning("Username already exists. Please choose a different username.")
            else:
                credentials[new_username] = new_password
                save_credentials(credentials)
                st.success("Sign-In successful! You can now login.")
                switch_page("login")
        else:
            st.error("Please enter a username and password")

# Login page
def login():
    st.title("Log-In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    def on_click_login():
        l(username, password)
    st.button("Log-In", on_click=on_click_login)

def l(username, password):
    credentials = load_credentials()
    if username in credentials and credentials[username] == password:
        st.session_state['logged_in'] = True
        switch_page("main")
    else:
        st.error("Invalid username or password")

# Main application page
def main_app():
    # Create the datasets directory if it doesn't exist
    os.makedirs('datasets', exist_ok=True)

    # Load dataframe from session state or file if available
    if st.session_state['df'] is None and os.path.exists('./datasets/dataset.csv'):
        st.session_state['df'] = pd.read_csv('./datasets/dataset.csv', index_col=None)
        st.session_state['df_original'] = st.session_state['df'].copy()  # Keep original dataframe
        st.session_state['file_name'] = 'dataset.csv'

    df = st.session_state['df']
    df_original = st.session_state['df_original']
    file_name = st.session_state['file_name']
    removed_features = st.session_state['removed_features']

    with st.sidebar:
        st.title("Innovator's AutoML Project")
        st.image('https://files.oaiusercontent.com/file-H8D2qKQfxfd1zrtTpZ67ir?se=2024-12-30T08%3A46%3A32Z&sp=r&sv=2024-08-04&sr=b&rscc=max-age%3D604800%2C%20immutable%2C%20private&rscd=attachment%3B%20filename%3Da6ee64be-afca-4c74-b533-dca2a32a3fc8.webp&sig=A7wt2DeNeurq%2Bo48TCbusoKds7kSBPHlZyMSzYobMvI%3D')
        
        choice = st.radio("Navigation", ["Data Ingestion", "Exploratory Data Analysis", "Data Transformation", "Modelling", "Download"])
        
    if choice == "Data Ingestion":
        st.title("Upload Your Dataset")
        file = st.file_uploader("Upload Your Dataset")
        if file:
            df = pd.read_csv(file, index_col=None)
            df.to_csv('./datasets/dataset.csv', index=None)
            st.session_state['df'] = df  # Store dataframe in session state
            st.session_state['df_original'] = df.copy()  # Keep original dataframe
            st.session_state['file_name'] = file.name  # Store file name in session state
            st.success(f"Dataset {file.name} uploaded successfully!")
            st.dataframe(df)

        # Display the uploaded file name if it exists
        if file_name:
            st.write(f"Uploaded file: {file_name}")
            if df is not None:
                st.write(f"Data Dimensions : ", df.shape)
                #st.write(f"Number of rows: {df.shape[0]}")
                #st.write(f"Number of columns: {df.shape[1]}")

    if choice == "Exploratory Data Analysis":
        if df is not None:
            st.title("Exploratory Data Analysis")
            profile_df = df.profile_report()
            st_profile_report(profile_df)
        else:
            st.warning("Please upload a dataset first.")
            
    if choice == "Data Transformation":
        if df is not None:
            st.title("Data Transformation")
            # Multi-select for columns to remove
            selected_columns = st.multiselect("Select columns to ignore", df.columns)
            if selected_columns:
                st.session_state['removed_features'].extend(selected_columns)
                st.session_state['df'] = st.session_state['df'].drop(columns=selected_columns)
                st.success(f"You have selected {len(selected_columns)} column(s) to ignore: {', '.join(selected_columns)}")
                df = st.session_state['df']
                st.write("Remaining columns:", df.columns.tolist())
                st.write("Removed columns:", st.session_state['removed_features'])
            else:
                st.write("No columns selected")

            # Dropdown to add back removed features
            if st.session_state['removed_features']:
                add_feature = st.selectbox("Select a column to add back", st.session_state['removed_features'])
                if st.button("Add Feature"):
                    st.session_state['removed_features'].remove(add_feature)
                    st.session_state['df'][add_feature] = st.session_state['df_original'][add_feature]
                    st.write("Remaining columns:", st.session_state['df'].columns.tolist())
                    st.write("Removed columns:", st.session_state['removed_features'])
            st.write("Final Shape:" , df.shape)
        else:
            st.warning("Please upload a dataset first.")
            
    if choice == "Modelling":
        if df is not None:
            st.title("Auto Train Model")

            # Update dataframe based on removed features
            df = st.session_state['df']  # Ensure df reflects the most recent state

            chosen_target = st.selectbox('Choose the Target Column:', df.columns)
            train_size = st.number_input('Enter the Training Size:', min_value=0.0, max_value=1.0, value=0.7, step=0.05, format="%.2f")
            test_size = 1 - train_size
            formatted_test_size = f"{test_size:.2f}"
            st.write("Testing Size : ", formatted_test_size)

            if st.button('Run Modelling'):
                st.title("Classification Models:")

                # Import PyCaret classification and regression modules with aliases
                from pycaret.classification import (
                    setup as classification_setup,
                    compare_models as compare_classification_models,
                    create_model,
                    tune_model,
                    pull,
                    plot_model,
                    save_model
                )
                from pycaret.regression import (
                    setup as regression_setup,
                    compare_models as compare_regression_models,
                    pull as pull_regression,
                    save_model as save_regression_model
                )

                # Make sure df is up-to-date
                df = st.session_state['df']

                # Classification Setup
                classification_setup(
                    df,
                    target=chosen_target,
                    ignore_features=st.session_state['removed_features'],
                    train_size=train_size,
                    fold=10,  # Cross-validation
                    fix_imbalance=True  # Handle class imbalance
                )
                st.dataframe(pull())

                # Compare all classification models
                best_model_classification = compare_classification_models()
                st.title("Classification Model Results: ")
                st.dataframe(pull())  # Show all model results
                save_model(best_model_classification, 'best_model_classification')

                # Train and tune Random Forest Model
                st.title("Random Forest Model")
                rf_model = create_model('rf')  # Random Forest
                tuned_rf_model = tune_model(rf_model)  # Tune RF Model
                st.write("Tuned Random Forest Model Results:")
                st.dataframe(pull())

                st.title("Confusion Matrix for Random Forest Model:")
                plot_model(tuned_rf_model, plot='confusion_matrix', display_format='streamlit')

                # Plot ROC Curve for Random Forest
                st.title("ROC Curve for Random Forest:")
                plot_model(tuned_rf_model, plot='auc', display_format='streamlit')

                # Train and tune KNN Model
                st.title("K-Nearest Neighbors Model")
                knn_model = create_model('knn')  # KNN
                tuned_knn_model = tune_model(knn_model)  # Tune KNN Model
                st.write("Tuned KNN Model Results:")
                st.dataframe(pull())

                st.title("Confusion Matrix for KNN Model:")
                plot_model(tuned_knn_model, plot='confusion_matrix', display_format='streamlit')

                # Plot ROC Curve for KNN
                st.title("ROC Curve for KNN:")
                plot_model(tuned_knn_model, plot='auc', display_format='streamlit')

                st.title("Regression Models:")

                # Regression Setup
                regression_setup(
                    df,
                    target=chosen_target,
                    ignore_features=st.session_state['removed_features'],
                    train_size=train_size
                )
                st.dataframe(pull_regression())  # Show regression setup results

                # Compare all regression models
                best_model_regression = compare_regression_models()
                st.dataframe(pull_regression())  # Show all model results
                save_regression_model(best_model_regression, 'best_model_regression')

        else:
            st.warning("Please upload a dataset first.")

    if choice == "Download":
        st.title("Hold Learnings: ")

        # Classification Model Download
        if os.path.exists('best_model_classification.pkl'):
            with open('best_model_classification.pkl', 'rb') as f:
                st.download_button('Download Best Classification Model', f, file_name="best_model_classification.pkl")
        else:
            st.warning("Please build a classification model first.")

        # Regression Model Download
        if os.path.exists('best_model_regression.pkl'):
            with open('best_model_regression.pkl', 'rb') as f:
                st.download_button('Download Best Regression Model', f, file_name="best_model_regression.pkl")
        else:
            st.warning("Please build a regression model first.")



# Determine which page to show
if st.session_state['logged_in']:
    main_app()
else:
    # Add a radio button for user to choose Sign In or Login
    page_choice = st.radio("Choose action", ["Sign In", "Login"])
    if page_choice == "Sign In":
        sign_in()
    else:
        login()
