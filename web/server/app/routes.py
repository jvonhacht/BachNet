from app import app
from bach_net import harmonize_json
from flask import render_template, url_for

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/harmonize', methods=['POST'])
def handle_harmonize(melody):
    return harmonize_json()
