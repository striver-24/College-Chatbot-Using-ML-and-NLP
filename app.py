from flask import Flask, render_template, request
import pickle
import json
from chatbot_core import EnhancedChatBot
import nltk

app = Flask(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load the trained model and vectorizer
with open('model/chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

# Initialize the enhanced chatbot
chatbot = EnhancedChatBot(model, vectorizer, intents)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    # Use client IP as session ID (or implement your own session management)
    session_id = request.remote_addr
    response = chatbot.get_response(user_input, session_id)
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5001)