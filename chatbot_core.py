import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import nltk
import json
import random
import re
from datetime import datetime, timedelta

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

class EnhancedChatBot:
    def __init__(self, model, vectorizer, intents_data):
        self.model = model
        self.vectorizer = vectorizer
        self.intents = intents_data['intents']
        self.lemmatizer = WordNetLemmatizer()
        self.response_history = defaultdict(list)  # Track responses by session
        self.conversation_memory = defaultdict(list)  # Store conversation history by session
        self.last_response_time = {}  # Track last response time by session
        self.memory_duration = timedelta(minutes=30)  # How long to remember conversation context
        
        # Create a mapping of tags to their intents for quick lookup
        self.tag_to_intent = {intent['tag']: intent for intent in self.intents}
        
    def preprocess_text(self, text):
        """Preprocess the input text for better understanding."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        
        return ' '.join(tokens)
    
    def get_intent(self, text):
        """Predict the intent of the input text."""
        preprocessed_text = self.preprocess_text(text)
        # Transform text using the pre-trained vectorizer
        X = self.vectorizer.transform([preprocessed_text])
        # Predict intent using the pre-trained model
        intent_idx = self.model.predict(X)[0]
        return self.intents[intent_idx]['tag']
    
    def get_context_similarity(self, text, session_id):
        """Calculate similarity between input and recent conversation context."""
        if not self.conversation_memory[session_id]:
            return 0
        
        # Get recent context
        recent_context = ' '.join([msg for msg in self.conversation_memory[session_id][-3:]])
        
        # Vectorize both the input and context
        texts = [self.preprocess_text(text), self.preprocess_text(recent_context)]
        vectors = self.vectorizer.transform(texts)
        
        # Calculate cosine similarity
        similarity = (vectors[0] * vectors[1].T).toarray()[0][0]
        return similarity
    
    def select_response(self, intent_tag, session_id):
        """Select an appropriate response based on intent and context."""
        intent = self.tag_to_intent.get(intent_tag)
        if not intent:
            return "I'm not sure how to respond to that."
        
        responses = intent['responses']
        if not responses:
            return "I don't have any responses for that topic."
        
        # Get response history for this session
        recent_responses = self.response_history[session_id]
        
        # Score each response based on various factors
        scored_responses = []
        for response in responses:
            score = 1.0
            
            # Penalize recently used responses
            if response in recent_responses:
                penalty = 0.5 ** (len(recent_responses) - recent_responses.index(response))
                score *= penalty
            
            # Add some randomness to prevent repetition
            score *= random.uniform(0.8, 1.2)
            
            scored_responses.append((score, response))
        
        # Select the highest scoring response
        selected_response = max(scored_responses, key=lambda x: x[0])[1]
        
        # Update response history
        self.response_history[session_id].append(selected_response)
        if len(self.response_history[session_id]) > 10:  # Keep last 10 responses
            self.response_history[session_id].pop(0)
        
        return selected_response
    
    def update_conversation_memory(self, text, session_id):
        """Update the conversation memory for the session."""
        current_time = datetime.now()
        
        # Clear old messages
        if session_id in self.last_response_time:
            time_diff = current_time - self.last_response_time[session_id]
            if time_diff > self.memory_duration:
                self.conversation_memory[session_id].clear()
        
        # Update memory and timestamp
        self.conversation_memory[session_id].append(text)
        self.last_response_time[session_id] = current_time
        
        # Keep only last 10 messages
        if len(self.conversation_memory[session_id]) > 10:
            self.conversation_memory[session_id].pop(0)
    
    def get_response(self, text, session_id='default'):
        """Generate a response to the input text."""
        try:
            # Update conversation memory
            self.update_conversation_memory(text, session_id)
            
            # Get intent
            intent_tag = self.get_intent(text)
            
            # Select and return appropriate response
            response = self.select_response(intent_tag, session_id)
            
            # Update conversation memory with response
            self.update_conversation_memory(response, session_id)
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now." 