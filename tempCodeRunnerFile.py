from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
with open('wine_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([[
        data['fixedAcidity'], data['volatileAcidity'], data['citricAcid'],
        data['residualSugar'], data['chlorides'], data['freeSulfurDioxide'],
        data['totalSulfurDioxide'], data['density'], data['pH'],
        data['sulphates'], data['alcohol']
    ]])

    prediction = model.predict(input_data)
    result = {
        'prediction': int(prediction[0])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
