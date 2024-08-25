from flask import Flask, request, jsonify
from model import process
app = Flask(__name__)

@app.route('/detect_spam_message', methods=['POST'])
def process_data():
    data = request.json
    msg = data.get('message')

    if msg:
        # Decode the Base64 encoded image
        try:
            # Classify the image
            type_message = process(msg)
            return jsonify({'type_message': type_message})
        except Exception as e:
            return jsonify({'error': 'Failed to decode image', 'message': str(e)}), 400
    else:
        return jsonify({'error': 'No message provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)
