import json

def get_feature_mapping(category,feature):
    return list(filter(lambda x:x["nameId"]==feature,feature_form_structure[category]))[0]["Data"]["Form"]


coef_types_weights = {
    'Normal' : lambda x : True if -1 <= x < 1 else False,
    'Poor': lambda x : True if -3 <= x < -1 else False,
    'Rich': lambda x : True if 1 <= x < 3 else False,
    'Very Poor': lambda x : True if -5 <= x < -3 else False,
    'Very Rich': lambda x : True if 3 <= x < 5 else False,
}

def classify_area(weight,coef_types_weights = coef_types_weights):
    return list({k:v(weight) for k,v in  coef_types_weights.items() if v(weight)})[0]

with open("./features_data/data.json",'r') as file:
    file_data = json.load(file)
    features_data = file_data["data"]
    counties_weights = features_data["counties_weights"]
    districts_weights = features_data["districts_weights"]
    cities_weights = features_data["cities_weights"]
    postcodeArea_weights = features_data["PostcodeArea_weights"]
    income_ByPostcode = features_data["income_ByPostcode"]

feature_form_structure = {
    'Geo-Zone': [
        {
            'Label': 'County',
            'nameId': "county",
            'Data' :  {
                'Description': 'Identify the County of the House.',
                "Form" : counties_weights
            }
        },
        {
            'Label': 'District',
            'nameId': "district",
            'Data' :  {
                'Description': 'Identify the District zone of the House.',
                "Form": districts_weights
            }
        },
        {
            'Label': 'City',
            'nameId': "city",
            'Data' :  {
                'Description': 'City of the property',
                'Form': cities_weights
            }
        },
        {
            'Label': 'Postcode Area',
            'nameId': "postcodeArea",
            'Data' :  {
                'Description': 'Area based on postcode of the property',
                'Form': postcodeArea_weights
            }
        },
    ],
    'Time': [
        {
            'Label': 'Age',
            'nameId': "oldNew",
            'Data' :  {
                'Description': 'Age of the property.',
                'Form': {
                    "Old": "Old",
                    "New": "New",
                }
            }
        },
        {
            'Label': 'Month',
            'nameId': "month",
            'Data' :  {
                'Description': 'Month of the purchase',
                'Form': {
                    "January": 1,
                    "February": 2,
                    "March": 3,
                    "April": 4,
                    "May": 5,
                    "June": 6,
                    "July": 7,
                    "August": 8,
                    "September": 9,
                    "October": 10,
                    "November": 11,
                    "December": 12,
                }
            }
        },
        {
            'Label': 'Duration',
            'nameId': "duration",
            'Data' :  {
                'Description': 'Type of the Allocation.',
                'Form': {
                    "Free Hold": "Freehold",
                    "Lease Hold": "Leasehold",
                }
            }
        },
    ],
    'Finance': [
        {
            'Label': 'Finance',
            'nameId': "ppdCategoryType",
            'Data' :  {
                'Description': 'type of Financemnt of the property.',
                'Form': {
                    "Standard": "StandardPrice",
                    "Additional": "AdditionalPrice",
                }
            }
        },
    ],
    'Property Details': [
        {
            'Label': 'Property Type',
            'nameId': "propertyType",
            'Data' :  {
                'Description': 'type of the property.',
                'Form': {
                    "Detached": "Detached",
                    "Semi-Detached": "SemiDetached",
                    "Terraced": "Terraced",
                    "Flat": "Flat",
                    "Other": "Other",
                }
            }
        },
    ],
}
