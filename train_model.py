import nltk
import json
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text, lemmatizer):
    """Preprocess the input text for better understanding."""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

# Prepare training data
X = []  # patterns
y = []  # intent indices
lemmatizer = WordNetLemmatizer()

for i, intent in enumerate(intents['intents']):
    for pattern in intent['patterns']:
        # Preprocess each pattern
        processed_pattern = preprocess_text(pattern, lemmatizer)
        X.append(processed_pattern)
        y.append(i)

# Create and fit the vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_vectorized, y)

# Save the model and vectorizer
with open('model/chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved successfully!") 