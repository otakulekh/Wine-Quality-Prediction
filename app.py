from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('wine_quality_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form (all 11 inputs)
    try:
        features = [
            float(request.form['fixed_acidity']),
            float(request.form['volatile_acidity']),
            float(request.form['citric_acid']),
            float(request.form['residual_sugar']),
            float(request.form['chlorides']),
            float(request.form['free_sulfur_dioxide']),
            float(request.form['total_sulfur_dioxide']),
            float(request.form['density']),
            float(request.form['pH']),
            float(request.form['sulphates']),
            float(request.form['alcohol'])
        ]
    except ValueError:
        return render_template('index.html', prediction_text='Invalid input, please enter numeric values for all fields.')
    
    # Convert features into a numpy array
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    # Return the result to the HTML template
    return render_template('index.html', prediction_text='Predicted Wine Quality: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
