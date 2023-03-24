from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the saved model
model = pickle.load(open('Customer_Churn_Prediction.pkl', 'rb'))

# Render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    gender = request.form['gender']
    seniorcitizen = request.form['seniorcitizen']
    partner = request.form['partner']
    dependents = request.form['dependents']
    tenure = request.form['tenure']
    phoneservice = request.form['phoneservice']
    multiplelines = request.form['multiplelines']
    internetservice = request.form['internetservice']
    onlinesecurity = request.form['onlinesecurity']
    onlinebackup = request.form['onlinebackup']
    deviceprotection = request.form['deviceprotection']
    techsupport = request.form['techsupport']
    streamingtv = request.form['streamingtv']
    streamingmovies = request.form['streamingmovies']
    contract = request.form['contract']
    paperlessbilling = request.form['paperlessbilling']
    paymentmethod = request.form['paymentmethod']
    monthlycharges = float(request.form['monthlycharges'])
    totalcharges = float(request.form['totalcharges'])
    # 19
    # Create a data frame from the input data
    data = {'gender': [gender],
            'Seniorcitizen': [seniorcitizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'Tenure': [tenure],
            'PhoneService': [phoneservice],
            'MultipleLines': [multiplelines],
            'InternetService': [internetservice],
            'OnlineSecurity': [onlinesecurity],
            'OnlineBackup': [onlinebackup],
            'DeviceProtection': [deviceprotection],
            'TechSupport': [techsupport],
            'StreamingTV': [streamingtv],
            'StreamingMovies': [streamingmovies],
            'Contract': [contract],
            'PaperlessBilling': [paperlessbilling],
            'PaymentMethod': [paymentmethod],
            'MonthlyCharges': [monthlycharges],
            'TotalCharges': [totalcharges]}
    df = pd.DataFrame(data)

    # Convert categorical variables into numerical variables #16 categorical features
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']
    le = LabelEncoder()
    for feature in categorical_features:
        df[feature] = le.fit_transform(df[feature])

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    # print(X)

    # Make the prediction using the loaded model
    prediction = model.predict(X)

    # Render the prediction result
    # print(prediction)
    if prediction[0] == 1:
        result = 'This customer is likely to churn.'
    else:
        result = 'This customer is unlikely to churn.'
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
