########## Flask ##############

# from flask import Flask, request,jsonify
# import joblib
# import numpy as np 

# app = Flask(__name__)

# model = joblib.load('model.joblib')

# @app.route('/predict',methods=['POST'])
# def predict():
#     data = request.json
#     prediction = model.predict(np.array(data['input']).reshape(1,-1))
#     return jsonify({'prediction':prediction.tolist()})


# if __name__ == '__main__':
#     app.run(debug=True,host='127.0.0.1')


########## Fastapi ##############

from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np 

app = FastAPI()


class Item(BaseModel):
    feature1: float
    feature2: float 
    feature3: float
    feature4: float 

@app.post("/items/")
async def predict(item: Item):
    model = joblib.load('model.joblib')
    prediction = model.predict(np.array([item.feature1, item.feature2, item.feature3, item.feature4]).reshape(1,-1))
    print("Prediction:", prediction)
    return dict({'prediction':str(prediction)})