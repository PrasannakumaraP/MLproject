"""
Application for model serving.
"""
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

