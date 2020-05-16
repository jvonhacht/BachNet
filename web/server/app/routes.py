from app import app
import app.performer as performer
from flask import request, jsonify
from numpy import nan

harmonizer = performer.Harmonizer()

@app.route('/')
def hello():
    return "Hello, world!"

@app.route('/harmonize', methods=['POST'])
def handle_harmonize():
    req = request.json
    melody = req['melody']
    melody = [[beat if beat is not None else nan] for beat in melody]
    required_length = 100
    melody = melody[:required_length] + [nan] * (required_length - len(melody)) # Crop and pad
    harmonization = harmonizer.harmonize(melody, req['overfit'])
    return jsonify({"harmonies":harmonization})
