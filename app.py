from flask import Flask, render_template, request,jsonify
import pickle
import pandas as pd
import numpy as np
import joblib 
from sklearn.preprocessing import StandardScaler


app = Flask(__name__, template_folder='template')
adaboost = pickle.load(open('adaboost_best_raw.pkl', 'rb'))
scaler = pickle.load(open('scaler_raw.pkl','rb'))

@app.route('/')
def home():
    return render_template("homepage.html") 

def get_data():
    NumberOfDeviceRegistered = request.form.get('NumberOfDeviceRegistered')
    NumberOfAddress = request.form.get('NumberOfAddress')
    CashbackAmount = request.form.get('CashbackAmount') 
    Tenure = request.form.get('Tenure')
    WarehouseToHome = request.form.get('WarehouseToHome')
    DaySinceLastOrder	 = request.form.get('DaySinceLastOrder')
    PreferredLoginDevice = request.form.get('PreferredLoginDevice')
    CityTier = request.form.get('CityTier')
    
    
    PreferredPaymentMode = request.form.get('PreferredPaymentMode')
    Gender = request.form.get('Gender')
    
    PreferedOrderCat = request.form.get('PreferedOrderCat')
    SatisfactionScore = request.form.get('SatisfactionScore')
    MaritalStatus = request.form.get('MaritalStatus')
    
    Complain = request.form.get('Complain')
    
   


    d_dict = {'NumberOfDeviceRegistered':[NumberOfDeviceRegistered],'NumberOfAddress':[NumberOfAddress],
              'CashbackAmount':[CashbackAmount],'Tenure': [Tenure],'WarehouseToHome': [WarehouseToHome],
              'DaySinceLastOrder':[DaySinceLastOrder],'PreferredLoginDevice_Computer': [0],'PreferredLoginDevice_Mobile Phone':[0],'CityTier_1': [0],'CityTier_2':[0],
              'CityTier_3':[0],'PreferredPaymentMode_Cash on Delivery':[0],
              'PreferredPaymentMode_Credit Card':[0],'PreferredPaymentMode_Debit Card':[0],
              'PreferredPaymentMode_E wallet':[0],'PreferredPaymentMode_UPI':[0],
                'Gender_Female': [0],'Gender_Male': [0],'PreferedOrderCat_Fashion':[0],
                'PreferedOrderCat_Grocery':[0],'PreferedOrderCat_Laptop & Accessory':[0],'PreferedOrderCat_Mobile Phone':[0],
                'PreferedOrderCat_Others':[0],'SatisfactionScore_1':[0],'SatisfactionScore_2':[0],'SatisfactionScore_3':[0],
                'SatisfactionScore_4':[0],'SatisfactionScore_5':[0],'MaritalStatus_Divorced':[0],'MaritalStatus_Married':[0],
                'MaritalStatus_Single':[0],'Complain_0':[0],'Complain_1':[0],
                }
    replace_list = [PreferredLoginDevice, CityTier, PreferredPaymentMode,Gender,PreferedOrderCat,SatisfactionScore,
                    MaritalStatus]

    for key, value in d_dict.items():
        if key in replace_list:
            d_dict[key] = 1


    return pd.DataFrame.from_dict(d_dict, orient='columns')

def feature_imp(model, data):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_30 = indices[:30]
    data = data.iloc[:, top_30]
    return data


@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()

    # Apply the loaded scaler to the data
    scaled_data = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Make predictions using the AdaBoost model
    prediction = adaboost.predict(scaled_data)

    # Map numeric class labels to human-readable labels
    class_labels = {0: 'Not Churn', 1: 'Churn'}
    prediction = [class_labels[label] for label in prediction]

    # Return the prediction to the template
    return render_template("results.html", prediction=prediction)

    


if __name__=="__main__":
    app.run(debug=True)