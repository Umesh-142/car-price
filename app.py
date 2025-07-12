import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model.pkl', 'rb'))

car_mapping = {'Toyota': 2, 'Honda': 1, 'Ford': 0}
model_mapping = {'Corolla': 1, 'Civic': 0, 'Focus': 2, 'Camry': 3, 'Fusion': 4, 'Accord': 5}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    try:
        car = data.get('car')
        model_name = data.get('model')
        year = int(data.get('year'))

        car_code = car_mapping.get(car, 0)
        model_code = model_mapping.get(model_name, 0)

        input_data = np.array([[car_code, model_code, year]])
        prediction = model.predict(input_data)
        return jsonify({'status': 'success', 'predicted_price': round(prediction[0], 2)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
