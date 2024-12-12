from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and the scaler
model = joblib.load('groundwater_model.joblib')
scaler = joblib.load('groundwater_scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    net_annual_gw = float(request.form['net_annual_gw'])
    total_usage = float(request.form['total_usage'])
    gw_future_irrigation = float(request.form['gw_future_irrigation'])
    
    user_input = pd.DataFrame([[net_annual_gw, total_usage, gw_future_irrigation]], 
                              columns=['Net annual groundwater availability', 'Total_Usage', 'Groundwater availability for future irrigation use'])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)
    prediction_text = prediction[0]

    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)