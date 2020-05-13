from app import app
from bach_net import harmonize_json

@app.route('/')
def hello():
    return "Hello, world!"

@app.route('/harmonize', methods=['POST'])
def handle_harmonize(melody):
    return harmonize_json()
