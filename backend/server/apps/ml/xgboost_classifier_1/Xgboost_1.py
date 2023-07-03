# file backend/server/apps/ml/income_classifier/random_forest.py
import joblib
import pandas as pd
import numpy as np
label=[1,2,3,5]
class XgbClassifier:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.model = joblib.load(path_to_artifacts + "XGboost_classifier.joblib")

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])         
        return input_data

    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)  # only one sample
            
            

            prediction= label[np.argmax(prediction)]

            

            
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction