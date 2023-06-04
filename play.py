from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

model = load_model('/home/ubuntu/Desktop/alice/DATA/model.h5')
with open('/home/ubuntu/Desktop/alice/DATA/vectorizer.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']

    message_vector = vectorizer.texts_to_sequences([message])

    message_vector = pad_sequences(message_vector, maxlen=2282)  # Mudado para 2282

    response_vector = model.predict(message_vector)

    return jsonify({'response': response_vector.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

