                                                           Customer Churn Prediction Web App
This repository contains a Flask-based web application and a Jupyter Notebook for predicting customer churn in a telecom dataset. The web app allows users to input customer attributes through an interactive interface and receive a prediction on whether the customer is likely to churn, along with the churn probability. The machine learning model is a pre-trained Random Forest classifier, with preprocessing handled by a scaler and encoder.


Overview
The Customer Churn Prediction Web App uses a Random Forest model to predict whether a telecom customer will churn based on 19 features, such as tenure, contract type, and monthly charges. The web interface, built with Flask and styled with Bootstrap and custom CSS, provides a responsive and visually appealing form for user input. The Jupyter Notebook (churn_prediction.ipynb) details the data exploration, preprocessing, and model training process using the Telco Customer Churn dataset.

Features
Interactive Web Interface: A form styled with Bootstrap and CSS animations for entering customer details.
Real-Time Predictions: Outputs "Churn" or "No Churn" with the probability of churn.
Comprehensive Inputs: Supports 19 features, including demographic, service, and billing information.
Pre-Trained Model: Random Forest classifier with SMOTE to handle imbalanced data.
Data Exploration: Includes visualizations (histograms, boxplots, heatmaps, countplots) in the Jupyter Notebook.
Responsive Design: Adapts to various screen sizes with smooth animations.
Dataset
The project uses the Telco Customer Churn dataset, which contains 7,043 records and 21 features (20 after dropping customerID). Key features include:

Demographic: gender, SeniorCitizen, Partner, Dependents
Service: PhoneService, InternetService, OnlineSecurity, etc.
Billing: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod
Target: Churn (Yes/No, encoded as 1/0)
The dataset is analyzed and preprocessed in churn_prediction.ipynb.

File Structure
text

Copy
customer-churn-prediction/
│
├── app.py                    # Flask application for serving the web app
├── templates/
│   └── index.html            # HTML template for the web interface
├── churn_prediction.ipynb    # Jupyter Notebook for data analysis and model training
├── best_models.pkl           # Pickled Random Forest model
├── scaler.pkl                # Pickled StandardScaler for numerical features
├── encoder.pkl               # Pickled LabelEncoder for categorical features
├── README.md                 # This file
Installation
Follow these steps to set up the project locally:

Clone the Repository:
bash

Copy
git clone https://github.com/Rohith1234-sarikela/customer-churn-prediction.git
cd customer-churn-prediction
Create a Virtual Environment (recommended):
bash

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies: Ensure Python 3.8+ is installed. Install the required packages:
bash

Copy
pip install flask pandas scikit-learn imblearn xgboost seaborn matplotlib jupyter
Verify Model Files: Ensure best_models.pkl, scaler.pkl, and encoder.pkl are in the root directory. These files are required for predictions.
Usage
Run the Flask Application:
bash

Copy
python app.py
The app will start at http://127.0.0.1:5000.
Access the Web Interface: Open a browser and navigate to http://127.0.0.1:5000. Fill out the form with customer details (e.g., gender, tenure, services).
View Prediction: Click "Predict" to see the churn prediction and probability displayed below the form.
Explore the Notebook: Run churn_prediction.ipynb to review the data analysis and model training:
bash

Copy
jupyter notebook churn_prediction.ipynb

The notebook includes:
Data loading and cleaning
Visualizations (distributions, correlations)
Preprocessing (encoding, scaling, SMOTE)
Model training with Random Forest and XGBoost
Model evaluation (accuracy, ROC-AUC, confusion matrix)

Model Details

Model: Random Forest Classifier (n_estimators=200, max_depth=None), selected via GridSearchCV.

Preprocessing:
Categorical features (e.g., gender, Contract) are encoded using LabelEncoder (saved in encoder.pkl).
Numerical features (tenure, MonthlyCharges, TotalCharges) are scaled using StandardScaler (saved in scaler.pkl).
Imbalanced data is handled with SMOTE to balance the Churn classes.

Performance (on test set):
Accuracy: 75.30%
ROC-AUC: 71.43%
Precision/Recall/F1 for Churn=1: 0.52/0.63/0.57

Training Process: Detailed in churn_prediction.ipynb, including hyperparameter tuning for Random Forest and XGBoost.

Dependencies
Python 3.8+
Flask
Pandas
Scikit-learn
Imbalanced-learn
XGBoost
Seaborn
Matplotlib
Jupyter
Install all dependencies with:

bash

Copy
pip install flask pandas scikit-learn imblearn xgboost seaborn matplotlib jupyter
