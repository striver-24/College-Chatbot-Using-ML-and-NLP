import nltk
import numpy as np
import random
import json
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
from collections import defaultdict

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

class ChatbotDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class EnhancedNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EnhancedNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l2(out)
        out = self.batch_norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.l3(out)
        return out

class EnhancedChatBot:
    def __init__(self, model, vectorizer, intents):
        self.model = model
        self.vectorizer = vectorizer
        self.intents = intents
        self.lemmatizer = WordNetLemmatizer()
        self.response_history = defaultdict(list)  # Track responses per intent
        self.last_response_time = defaultdict(float)  # Track when each response was last used
        self.context = []  # Store conversation context
    
    def preprocess(self, text):
        # Tokenize and lemmatize
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join([self.lemmatizer.lemmatize(token) for token in tokens])
    
    def get_intent(self, text):
        # Preprocess and predict intent
        processed_text = self.preprocess(text)
        X = self.vectorizer.transform([processed_text])
        output = self.model.predict(X)
        return output[0]
    
    def select_response(self, intent_tag, text):
        # Find the matching intent
        matching_intent = None
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                matching_intent = intent
                break
        
        if not matching_intent:
            return "I'm not sure how to respond to that."
        
        responses = matching_intent['responses']
        current_time = time.time()
        
        # Score each response based on multiple factors
        response_scores = []
        for idx, response in enumerate(responses):
            score = 0.0
            
            # Penalize recently used responses
            last_used = self.last_response_time[f"{intent_tag}_{idx}"]
            time_diff = current_time - last_used
            recency_score = min(1.0, time_diff / 3600)  # Max penalty for responses used within last hour
            score += recency_score * 0.4
            
            # Penalize responses that were used recently for this intent
            if response in self.response_history[intent_tag]:
                score -= 0.3
            
            # Add some randomness
            score += random.uniform(0, 0.3)
            
            response_scores.append(score)
        
        # Select response with highest score
        best_idx = np.argmax(response_scores)
        selected_response = responses[best_idx]
        
        # Update tracking
        self.last_response_time[f"{intent_tag}_{best_idx}"] = current_time
        self.response_history[intent_tag].append(selected_response)
        
        # Keep only last 3 responses in history
        if len(self.response_history[intent_tag]) > 3:
            self.response_history[intent_tag].pop(0)
        
        return selected_response
    
    def get_response(self, text):
        # Update context
        self.context.append(text)
        if len(self.context) > 5:
            self.context.pop(0)
        
        # Get intent and select response
        intent = self.get_intent(text)
        response = self.select_response(intent, text)
        
        return response

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            total_accuracy += accuracy(outputs, targets) * inputs.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_accuracy = total_accuracy / len(train_loader.dataset)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

def accuracy(predictions, targets):
    predicted_labels = torch.argmax(predictions, dim=1)
    true_labels = torch.argmax(targets, dim=1)
    correct = (predicted_labels == true_labels).sum().item()
    total = targets.size(0)
    return correct / total 