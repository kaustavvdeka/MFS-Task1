import os
import random
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to save plots without opening window
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import KNOWLEDGE_BASE and other helper functions from chatbot.py
from chatbot import KNOWLEDGE_BASE, preprocess, regex_match

# ==========================================
# DOWNLOAD NLTK RESOURCES
# ==========================================
print("Checking NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print("All NLTK resources ready!")

# ==========================================
# ADVANCED NLP PREPROCESSING
# ==========================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'very', 'too'}
# Add common conversational verbs and helper words to stop words to avoid false keyword overlaps
stop_words.update({'get', 'want', 'would', 'could', 'please', 'like', 'go', 'take', 'make'})

def advanced_preprocess(text):
    '''Full NLP preprocessing: tokenise, lemmatise, remove stop words.'''
    text = text.lower()
    tokens = word_tokenize(text)
    # Remove punctuation tokens (keep only words consisting of alphabetical characters)
    tokens = [t for t in tokens if t.isalpha()]
    # Remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    # Lemmatise tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# ==========================================
# GLOBAL VARIABLES FOR MODEL
# ==========================================
tfidf = None
lr_clf = None
le = None
X_tfidf = None
y = []
X_raw = []
X_proc = []

# Paths for serialized models
MODEL_PATH = 'chatbot_tfidf.pkl'
ENCODER_PATH = 'chatbot_encoder.pkl'
LABELS_PATH = 'chatbot_labels.pkl'

# ==========================================
# SAVE & LOAD FUNCTIONS
# ==========================================
def save_model():
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(lr_clf, f)
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump((X_tfidf, y, X_raw, le), f)
    print('Model saved!')

def load_model():
    global tfidf, lr_clf, X_tfidf, y, X_raw, le
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(LABELS_PATH):
        with open(MODEL_PATH, 'rb') as f:
            tfidf = pickle.load(f)
        with open(ENCODER_PATH, 'rb') as f:
            lr_clf = pickle.load(f)
        with open(LABELS_PATH, 'rb') as f:
            X_tfidf, y, X_raw, le = pickle.load(f)
        print('Model loaded from disk!')
        return True
    return False

# ==========================================
# TRAINING PIPELINE & EVALUATION
# ==========================================
def train_model():
    global tfidf, lr_clf, le, X_tfidf, y, X_raw, X_proc
    print("Building training dataset from knowledge base...")
    
    X_raw = []
    X_proc = []
    y = []
    
    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == 'unknown':
            continue  # Skip fallback entry for training dataset
        for pattern in entry['patterns']:
            X_raw.append(pattern)
            X_proc.append(advanced_preprocess(pattern))
            y.append(entry['tag'])
            
    print(f'Training samples: {len(X_proc)}')
    print(f'Unique tags     : {len(set(y))}')
    
    # Train TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=1,            # do not ignore rare words due to small dataset
        analyzer='word'
    )
    X_tfidf = tfidf.fit_transform(X_proc)
    print(f'Vocabulary size : {len(tfidf.vocabulary_)}')
    print(f'TF-IDF shape    : {X_tfidf.shape}')
    
    # Encode tags
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/Test Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_tfidf, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Logistic Regression
    lr_clf = LogisticRegression(max_iter=500, random_state=42)
    lr_clf.fit(X_tr, y_tr)
    
    # Evaluate
    lr_preds = lr_clf.predict(X_te)
    print(f'\nLogistic Regression Accuracy: {accuracy_score(y_te, lr_preds):.4f}')
    print('\nClassification Report:')
    print(classification_report(y_te, lr_preds, target_names=le.classes_))
    
    # Generate and save confusion matrix
    print("Generating confusion matrix plot...")
    y_pred_labels = le.inverse_transform(lr_preds)
    y_true_labels = le.inverse_transform(y_te)
    
    unique_tags = sorted(list(set(y)))
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=unique_tags)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_tags, yticklabels=unique_tags,
                linewidths=0.5, annot_kws={'size':11})
    plt.title('Chatbot Intent Classification — Confusion Matrix',
              fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('True Intent', fontsize=11)
    plt.xlabel('Predicted Intent', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('chatbot_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved as 'chatbot_confusion_matrix.png'")
    
    # Save the model
    save_model()

# ==========================================
# COSINE SIMILARITY INTENT DETECTION
# ==========================================
def predict_intent_cosine(user_input, threshold=0.25):
    '''
    Predict intent using TF-IDF cosine similarity.
    threshold: minimum similarity score to accept a match
    '''
    processed = advanced_preprocess(user_input)
    user_vector = tfidf.transform([processed])
    
    # Calculate similarity between user input and ALL training patterns
    similarities = cosine_similarity(user_vector, X_tfidf)[0]
    
    # Find the most similar pattern
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    best_tag = y[best_idx]
    
    # Print debug information
    print(f' [Debug] Best match: "{X_raw[best_idx]}" | Score: {best_score:.3f} | Tag: {best_tag}')
    
    if best_score < threshold:
        best_tag = 'unknown'
        
    # Return a random response for the detected intent
    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == best_tag:
            return best_tag, random.choice(entry['responses']), best_score
            
    # Fallback response for 'unknown'
    for entry in KNOWLEDGE_BASE:
        if entry['tag'] == 'unknown':
            return 'unknown', random.choice(entry['responses']), 0.0

# ==========================================
# NLP CHATBOT LOOP
# ==========================================
def nlp_chat():
    print('=' * 60)
    print(' CollegeBot NLP Edition — AI-Powered Assistant')
    print(' Main Flow Services and Technologies Pvt. Ltd.')
    print('=' * 60)
    print('Ask me anything about the college! Type "bye" to exit.\n')
    
    history = []
    turn = 0
    
    while True:
        user_input = input('You: ').strip()
        if not user_input:
            continue
            
        turn += 1
        history.append(user_input)
        
        # Use NLP intent detection
        tag, response, score = predict_intent_cosine(user_input)
        
        # If confidence is very low, try the old rule-based system
        if score < 0.15:
            tag2, response = regex_match(user_input)
            tag = tag2
            print(f' [Switched to rule-based: tag={tag2}]')
        else:
            print(f' [NLP detected intent: {tag}, confidence: {score:.2f}]')
            
        print(f'Bot: {response}\n')
        
        # Exit if farewell detected
        if tag == 'farewell':
            print(f'Session complete. Total turns: {turn}')
            break

if __name__ == '__main__':
    # Load model if it exists, otherwise train it
    if not load_model():
        print("No pre-trained model files found. Training now...")
        train_model()
    nlp_chat()
