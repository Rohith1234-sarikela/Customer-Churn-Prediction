from flask import Flask, render_template, request
import pandas as pd 
import pickle

# Load model, scaler, and encoder
with open("best_models.pkl", "rb") as file:
    loaded_model = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    loaded_scaler = pickle.load(file)
with open("encoder.pkl", "rb") as file:
    loaded_encoder = pickle.load(file) 

app = Flask(__name__)

def make_prediction(input_data):
    # Convert input data into a DataFrame
    input_dataset = pd.DataFrame([input_data])

    # Apply transformations for categorical variables using the encoder
    for col, encoder in loaded_encoder.items():
        if col in input_dataset.columns:
            input_dataset[col] = encoder.transform(input_dataset[col])

    # Scale numerical columns
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    input_dataset[numerical_cols] = loaded_scaler.transform(input_dataset[numerical_cols])

    # Make prediction using the model
    prediction = loaded_model.predict(input_dataset)
    probability = loaded_model.predict_proba(input_dataset)[0, 1]

    return "Churn" if prediction == 1 else "No Churn", round(probability * 100, 2)

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        input_data = {
            'gender': request.form.get('gender'),
            'SeniorCitizen': int(request.form.get('SeniorCitizen')),
            'Partner': request.form.get('Partner'),
            'Dependents': request.form.get('Dependents'),
            'tenure': float(request.form.get('tenure')),
            'PhoneService': request.form.get('PhoneService'),
            'MultipleLines': request.form.get('MultipleLines'),
            'InternetService': request.form.get('InternetService'),
            'OnlineSecurity': request.form.get('OnlineSecurity'),
            'OnlineBackup': request.form.get('OnlineBackup'),
            'DeviceProtection': request.form.get('DeviceProtection'),
            'TechSupport': request.form.get('TechSupport'),
            'StreamingTV': request.form.get('StreamingTV'),
            'StreamingMovies': request.form.get('StreamingMovies'),
            'Contract': request.form.get('Contract'),
            'PaperlessBilling': request.form.get('PaperlessBilling'),
            'PaymentMethod': request.form.get('PaymentMethod'),
            'MonthlyCharges': float(request.form.get('MonthlyCharges')),
            'TotalCharges': float(request.form.get('TotalCharges'))
        }

        prediction, probability = make_prediction(input_data)

    return render_template("index.html", prediction=prediction, probability=probability)


if __name__ == "__main__":
    app.run(debug=True)
