import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from features_data.features import feature_form_structure, get_feature_mapping, income_ByPostcode, classify_area

app = Flask(__name__)

# upload model & features data (train_r2 = 0.79, test_r2 = 0.77)
model_name = 'model_xgboost_withIncomeArea'
model_dir = f'./models/{model_name}'
label_encoders_dir = f'{model_dir}/label-encoders'
onehot_encoder_dir = f'{model_dir}/onhot-encoders'

# model
model = pickle.load(open(f'{model_dir}/model.sav', 'rb'))

# one hot encoder
Month_ohe = pickle.load(open(onehot_encoder_dir+"/Month_ohe", 'rb'))
PropertyType_ohe = pickle.load(open(onehot_encoder_dir+"/PropertyType_ohe", 'rb'))

# duration encoder
Duration_encoder = LabelEncoder()
Duration_encoder.classes_ = np.load(label_encoders_dir+"/Duration.npy",allow_pickle=True)
# OldNew encoder
OldNew_encoder = LabelEncoder()
OldNew_encoder.classes_ = np.load(label_encoders_dir+"/OldNew.npy",allow_pickle=True)
# PDCategoryType encoder
PPDCategoryType_encoder = LabelEncoder()
PPDCategoryType_encoder.classes_ = np.load(label_encoders_dir+"/PPDCategoryType.npy",allow_pickle=True)


# Features values
months_data = get_feature_mapping("Time","month")
oldNew_data = get_feature_mapping("Time","oldNew")
duration_data = get_feature_mapping("Time","duration")
ppdCategoryType_data = get_feature_mapping("Finance","ppdCategoryType")
propertyType_data = get_feature_mapping("Property Details","propertyType")
counties_weights = get_feature_mapping("Geo-Zone","county")
districts_weights = get_feature_mapping("Geo-Zone","district")
cities_weights = get_feature_mapping("Geo-Zone","city")
postcodeArea_weights = get_feature_mapping("Geo-Zone","postcodeArea")




@app.route('/')
def home():
    return render_template('index.html',feature_form_structure=feature_form_structure)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values from the form
    data = {
        "county" : request.form['county'],
        "city" : request.form['city'],
        "district" : request.form['district'],
        "postcodeArea" : request.form['postcodeArea'],
        "propertyType" : request.form['propertyType'],
        "duration" : request.form['duration'],
        "oldNew" : request.form['oldNew'],
        "ppdCategoryType" : request.form['ppdCategoryType'],
        "month" : request.form['month']
    }
    
    # Encode variables
    feature_oldnew =  OldNew_encoder.transform([oldNew_data[data["oldNew"]]])
    feature_PPDCategoryType =  PPDCategoryType_encoder.transform([ppdCategoryType_data[data["ppdCategoryType"]]])
    feature_Duration =  Duration_encoder.transform([duration_data[data["duration"]]])
    feature_PropertyType = PropertyType_ohe.transform([[propertyType_data[data["propertyType"]]]])
    feature_Month = Month_ohe.transform([[months_data[data["month"]]]])
    
    # Coverting to Features
    features = [
        *feature_oldnew,
        *feature_Duration,
        *feature_PPDCategoryType,
        *feature_Month[0],
        *[
            counties_weights[data["county"]],
            districts_weights[data["district"]],
            cities_weights[data["city"]],
            postcodeArea_weights[data["postcodeArea"]]
        ],
        *feature_PropertyType[0],
        income_ByPostcode[data["postcodeArea"]]
    ]

    # add more info
    extra_data = {
        "areaIncome" : income_ByPostcode[data["postcodeArea"]],
        "county": classify_area(weight=counties_weights[data["county"]]),
        "city": classify_area(weight=cities_weights[data["city"]]),
        "district": classify_area(weight=districts_weights[data["district"]]),
        "postcodeArea": classify_area(weight=postcodeArea_weights[data["postcodeArea"]]),
    }


    # house price prediction
    prediction = model.predict([features])
    expected_price = np.expm1(prediction)[0]

    return render_template('result.html', data=data, extra_data=extra_data ,prediction=expected_price)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port)