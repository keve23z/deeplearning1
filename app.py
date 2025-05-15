# from flask import Flask, request, jsonify
# from infer import generate_answer

# app = Flask(__name__)

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.json.get('question')
#     response = generate_answer(user_input)
#     return jsonify({"answer": response})

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify, render_template
from infer import generate_answer

app = Flask(__name__)

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route for chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question')
    response = generate_answer(user_input)
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(debug=True)
