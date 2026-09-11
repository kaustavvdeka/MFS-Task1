CollegeBot — Rule-Based & NLP-Enhanced AI Chatbot

A complete Python chatbot project demonstrating two approaches to conversational intent detection:

Rule-Based Chatbot using preprocessing, keyword overlap, regular expressions, and a predefined knowledge base.

NLP-Enhanced Chatbot using tokenization, stop-word removal, lemmatization, TF-IDF, cosine similarity, label encoding, Logistic Regression, model serialization, and evaluation.

The project is designed as a college information assistant that can answer questions about admissions, fees, courses, hostel/accommodation, placements, greetings, and other common queries.

Table of Contents

1. Project Overview

2. Learning Objectives

3. Project Architecture

4. Project Structure

5. Core Concepts

Artificial Intelligence

Chatbots

NLP

Intent Classification

Rule-Based Systems

Machine Learning

Text Representation

6. Knowledge Base

7. Text Preprocessing

8. Rule-Based Matching

9. Regular Expressions

10. NLP Preprocessing

11. Tokenization

12. Stop Words

13. Lemmatization

14. TF-IDF

15. N-Grams

16. Cosine Similarity

17. Label Encoding

18. Train-Test Split

19. Logistic Regression

20. Model Evaluation

21. Confusion Matrix

22. Hybrid Intent Detection

23. Confidence Thresholds

24. Model Serialization

25. Conversation Loop

26. Python Libraries

27. Library Syntax Reference

28. Installation

29. Running the Project

30. Training Pipeline

31. Verification

32. Important Python Syntax

33. Mathematical Theory

34. End-to-End Data Flow

35. Rule-Based vs NLP Comparison

36. Advantages

37. Limitations

38. Improvements

39. Common Errors

40. Viva/Interview Questions

41. Conclusion

1. Project Overview

What is CollegeBot?

CollegeBot is a text-based conversational assistant that identifies the intent behind a user's message and returns an appropriate response.

For example:

User: What are the fees for engineering?

Detected intent: fees

Bot: Fee structure varies by course...

Another example:

User: I need information about accommodation.

Detected intent: hostel

Bot: Yes! Separate hostels are available...

The project contains two generations of chatbot logic.

Task 1 — Rule-Based Chatbot

The first implementation uses:

predefined patterns

keyword matching

set intersection

regular expressions

random responses

fallback handling

Task 2 — NLP-Enhanced Chatbot

The second implementation uses:

NLTK

tokenization

stop-word removal

lemmatization

TF-IDF

unigrams and bigrams

cosine similarity

Logistic Regression

LabelEncoder

train/test splitting

classification metrics

confusion matrix

pickle model persistence

hybrid fallback logic

2. Learning Objectives

This project demonstrates the complete basic NLP/ML pipeline:

Raw User Input
      ↓
Text Cleaning
      ↓
Tokenization
      ↓
Stop-Word Removal
      ↓
Lemmatization
      ↓
Numerical Representation
      ↓
TF-IDF
      ↓
Intent Matching / Classification
      ↓
Confidence Check
      ↓
Response Selection
      ↓
Bot Response

After studying this project, you should understand:

what NLP is

how a chatbot detects intent

how rule-based NLP works

how regular expressions work

what tokenization means

what stop words are

stemming vs lemmatization

TF-IDF

n-grams

cosine similarity

supervised classification

Logistic Regression

label encoding

train/test split

accuracy, precision, recall and F1-score

confusion matrices

model persistence with pickle

hybrid AI systems

3. Project Architecture

The system can be viewed as two layers.

Rule-Based Layer

User Input
   ↓
Preprocess
   ↓
Regex Matching
   ↓
If matched → Intent
   ↓
Otherwise
   ↓
Keyword Overlap
   ↓
Intent
   ↓
Response

NLP Layer

User Input
   ↓
Advanced Preprocessing
   ↓
Tokenization
   ↓
Stop-word Removal
   ↓
Lemmatization
   ↓
TF-IDF Vector
   ↓
Cosine Similarity
   ↓
Best Training Pattern
   ↓
Intent
   ↓
Response

The project also trains a Logistic Regression classifier for supervised evaluation:

Training Patterns
      ↓
NLP Preprocessing
      ↓
TF-IDF
      ↓
Label Encoding
      ↓
Train/Test Split
      ↓
Logistic Regression
      ↓
Prediction
      ↓
Evaluation

4. Project Structure

MFS-Task1-main/
│
├── README.md
├── chatbot.py
├── nlp_chatbot.py
├── verify_nlp.py
├── chatbot_project.ipynb
├── generate_report.py
├── report.md
├── report.pdf
└── .gitignore

chatbot.py

Contains the rule-based chatbot.

Main components:

KNOWLEDGE_BASE

preprocess()

find_best_match()

REGEX_PATTERNS

regex_match()

chat()

nlp_chatbot.py

Contains the NLP and machine-learning implementation.

Main components:

NLTK resource setup

advanced preprocessing

TF-IDF

cosine similarity

Logistic Regression

LabelEncoder

model saving/loading

evaluation

confusion matrix

NLP conversation loop

verify_nlp.py

Tests the trained chatbot using new phrasings.

chatbot_project.ipynb

Jupyter Notebook version of the project for experimentation and demonstration.

generate_report.py

Generates a PDF report using the FPDF library.

report.md

Project report in Markdown format.

5. Core Concepts

Artificial Intelligence

Artificial Intelligence (AI) is the field of computer science concerned with building systems that can perform tasks normally requiring human intelligence.

Examples:

language understanding

image recognition

recommendation

decision making

speech recognition

planning

A chatbot is an AI application when it can interpret user input and generate or select an appropriate response.

Chatbots

A chatbot is software that communicates with users using natural language.

Two major types are:

Rule-Based Chatbot

Uses manually defined rules.

IF input contains "fees"
THEN return fee information

Advantages:

simple

predictable

fast

easy to debug

Disadvantages:

cannot generalize well

depends on manually written rules

weak handling of synonyms and paraphrases

NLP/ML Chatbot

Uses NLP and machine learning to map language to intents.

"I need information about accommodation"
        ↓
NLP preprocessing
        ↓
TF-IDF representation
        ↓
Similarity
        ↓
hostel intent

NLP

Natural Language Processing is a branch of AI that enables computers to process human language.

Typical NLP operations:

Text
 ↓
Normalization
 ↓
Tokenization
 ↓
Stop-word removal
 ↓
Stemming/Lemmatization
 ↓
Feature extraction
 ↓
Machine Learning

NLP applications include:

chatbots

sentiment analysis

translation

search

text classification

spam detection

summarization

question answering

Intent Classification

An intent represents what the user wants.

Example:

User input

Intent

"Hi"

greeting

"How can I apply?"

admission

"What is the fee?"

fees

"What programs are available?"

courses

"Do you have a hostel?"

hostel

"What jobs can I get?"

placements

"Bye"

farewell

The goal of the model is:

Text → Intent

The response system then performs:

Intent → Response

Rule-Based Systems

A rule-based system uses explicitly written conditions.

Example:

if "fee" in user_input:
    return "fees"

This project improves this basic approach using regular expressions and keyword overlap.

Machine Learning

Machine Learning allows a computer to learn patterns from examples instead of requiring every decision to be explicitly programmed.

In this project:

Training examples
        ↓
TF-IDF features
        ↓
Logistic Regression
        ↓
Intent classifier

This is supervised learning because every training example has a known intent label.

Text Representation

Machine learning algorithms cannot directly understand text.

For example:

"what are the fees"

must be converted into numbers.

TF-IDF performs this transformation:

Text → numerical vector

The vector can then be processed mathematically.

6. Knowledge Base

The knowledge base is defined in chatbot.py.

Basic structure:

{
    'tag': 'fees',
    'patterns': [
        'what are the fees',
        'how much does it cost',
        'fee structure'
    ],
    'responses': [
        'Fee structure varies by course...'
    ]
}

Each entry contains:

tag

The intent name.

'tag': 'fees'

patterns

Example sentences associated with the intent.

'patterns': [
    'what are the fees',
    'fee structure',
    'course fee'
]

responses

Possible bot answers.

'responses': [
    'Fee structure varies by course...'
]

Multiple responses are used so that the chatbot does not always return exactly the same sentence.

7. Text Preprocessing

Text preprocessing converts raw text into a normalized form.

The basic preprocessing function is:

def preprocess(text):
    text = text.lower()
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )
    text = text.strip()
    text = ' '.join(text.split())
    return text

Step 1 — Lowercase

text = text.lower()

Example:

"HELLO COLLEGE!"

becomes:

"hello college!"

This prevents:

"Hello"

and

"hello"

from being treated as different strings.

Step 2 — Remove punctuation

text.translate(str.maketrans('', '', string.punctuation))

Example:

"Hello!"

becomes:

"Hello"

Step 3 — Remove surrounding spaces

text.strip()

Step 4 — Normalize repeated spaces

' '.join(text.split())

Example:

"hello     college"

becomes:

"hello college"

8. Rule-Based Matching

The keyword matching engine is:

def find_best_match(user_input):

The input is converted into a set:

user_words = set(cleaned.split())

For a sentence:

"what is the course fee"

the set could be:

{
    "what",
    "is",
    "the",
    "course",
    "fee"
}

A pattern is also converted into a set.

The overlap is calculated using:

overlap = len(user_words & pattern_words)

The & operator performs set intersection.

Example:

A = {"course", "fee", "college"}
B = {"course", "fee"}

A & B

Result:

{"course", "fee"}

Therefore:

overlap = 2

The pattern with the highest overlap becomes the best match.

9. Regular Expressions

Regular expressions, commonly called regex, are patterns used to search text.

Python provides regex through:

import re

Example:

re.search(r'\b(fee|fees|cost)\b', text)

This can match:

fee
fees
cost

Word Boundary

The expression:

\b

represents a word boundary.

For example:

\bfee\b

matches:

fee

but does not intentionally match the same character sequence inside a larger word such as:

coffee

OR operator

Regex:

fee|fees|cost

means:

fee OR fees OR cost

Grouping

(fee|fees|cost)

groups alternatives.

Whitespace

This pattern:

see\s+you

matches:

see you
see   you

because \s+ means one or more whitespace characters.

Regex in this project

The chatbot first tries regex:

for pattern, tag in REGEX_PATTERNS:
    if re.search(pattern, cleaned):
        ...

If no regex rule matches, it falls back to keyword overlap.

10. NLP Preprocessing

The advanced preprocessing pipeline is:

def advanced_preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

Pipeline:

Raw text
 ↓
Lowercase
 ↓
Tokenization
 ↓
Alphabetic filtering
 ↓
Stop-word removal
 ↓
Lemmatization
 ↓
Processed text

11. Tokenization

Tokenization divides text into smaller units called tokens.

Example:

"I want to study computer science"

can become:

[
    "I",
    "want",
    "to",
    "study",
    "computer",
    "science"
]

The project uses:

from nltk.tokenize import word_tokenize

Syntax:

tokens = word_tokenize(text)

12. Stop Words

Stop words are common words that often contribute little to text classification.

Examples:

the
is
a
an
to
of
in
on

The project uses:

from nltk.corpus import stopwords

Load English stop words:

stopwords.words('english')

Convert to a set:

stop_words = set(stopwords.words('english'))

A set gives efficient membership testing.

Why retain negation?

This project intentionally retains:

not
no
never
very
too

This is important because removing negation can change meaning.

For example:

"I do not want hostel"

is different from:

"I want hostel"

13. Lemmatization

Lemmatization converts words to a meaningful base form.

Example:

studies → study

Depending on linguistic information and the lemmatizer:

running → running/run

The project uses:

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

Syntax:

lemmatizer.lemmatize(word)

Example:

lemmatizer.lemmatize("cars")

may produce:

car

Lemmatization generally produces linguistically meaningful roots.

14. TF-IDF

TF-IDF stands for:

Term Frequency–Inverse Document Frequency

It converts text into numerical vectors while reducing the importance of very common words.

The project uses:

from sklearn.feature_extraction.text import TfidfVectorizer

Create vectorizer:

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    analyzer='word'
)

Train the vectorizer:

X_tfidf = tfidf.fit_transform(X_proc)

Term Frequency

Term Frequency measures how frequently a term appears in a document.

A basic formulation is:

[
TF(t,d) =
\frac{\text{number of occurrences of }t\text{ in }d}
{\text{total number of terms in }d}
]

Inverse Document Frequency

IDF reduces the weight of words appearing in many documents.

A common formulation is:

[
IDF(t)=\log\left(\frac{N+1}{df(t)+1}\right)+1
]

where:

N = number of documents

df(t) = number of documents containing term t

TF-IDF

[
TFIDF(t,d)=TF(t,d)\times IDF(t)
]

A word that is frequent in one document but rare across the dataset gets a relatively high weight.

Example

Suppose:

Document 1: fee structure college
Document 2: course fee college
Document 3: hostel college

college appears in many documents, so its IDF is relatively low.

hostel appears in fewer documents, so it can be more informative for identifying the hostel intent.

15. N-Grams

An n-gram is a sequence of n consecutive tokens.

Unigram

One word:

fee

Bigram

Two words:

fee structure

Trigram

Three words:

what are fees

The project uses:

ngram_range=(1, 2)

This means:

unigrams + bigrams

Examples:

course
fee
course fee
fee structure

This is useful because phrases can carry more meaning than individual words.

16. Cosine Similarity

Cosine similarity measures the similarity between two vectors.

Formula:

[
\cos(\theta)=
\frac{A\cdot B}
{|A||B|}
]

where:

A = vector 1

B = vector 2

A · B = dot product

||A|| = magnitude of A

||B|| = magnitude of B

The value is generally interpreted as:

1     → very similar direction
0     → little/no similarity

The project uses:

from sklearn.metrics.pairwise import cosine_similarity

Syntax:

similarities = cosine_similarity(
    user_vector,
    X_tfidf
)[0]

Then:

best_idx = similarities.argmax()

gets the index of the highest similarity.

17. Label Encoding

Machine learning models commonly work with numeric labels.

The intents are strings:

admission
fees
courses
hostel
placements

The project converts them into numerical labels using:

from sklearn.preprocessing import LabelEncoder

Create:

le = LabelEncoder()

Fit:

y_encoded = le.fit_transform(y)

Convert back:

le.inverse_transform(predictions)

Example conceptually:

admission  → 0
courses    → 1
fees       → 2
hostel     → 3

The exact numeric assignments depend on the encoder's fitted class ordering.

18. Train-Test Split

A dataset should be divided into training and testing portions.

The project uses:

from sklearn.model_selection import train_test_split

Syntax:

X_tr, X_te, y_tr, y_te = train_test_split(
    X_tfidf,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

test_size=0.2

20% of the data is used for testing.

Approximately:

80% → training
20% → testing

random_state=42

Makes the split reproducible.

stratify=y_encoded

Attempts to preserve class proportions between train and test sets.

19. Logistic Regression

Despite its name, Logistic Regression is a classification algorithm.

It estimates class probabilities using a logistic function.

For binary classification:

[
P(y=1|x)=
\frac{1}{1+e^{-z}}
]

where:

[
z=w^Tx+b
]

For multiclass classification, implementations such as scikit-learn extend the idea to multiple classes.

The project uses:

from sklearn.linear_model import LogisticRegression

Create model:

lr_clf = LogisticRegression(
    max_iter=500,
    random_state=42
)

Train:

lr_clf.fit(X_tr, y_tr)

Predict:

predictions = lr_clf.predict(X_te)

Why Logistic Regression works well here

The input is a sparse TF-IDF feature matrix.

Logistic Regression is often a strong baseline for text classification because:

it is relatively simple

it is computationally efficient

it works well with sparse features

it provides a strong baseline for small/medium text datasets

20. Model Evaluation

The project uses:

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

Accuracy

Accuracy measures the proportion of correct predictions.

[
Accuracy =
\frac{Correct\ Predictions}
{Total\ Predictions}
]

Python:

accuracy_score(y_true, y_pred)

Example:

70 correct out of 100

means:

Accuracy = 70%

Precision

Precision answers:

Of the examples predicted as a class, how many were actually that class?

[
Precision =
\frac{TP}{TP+FP}
]

where:

TP = True Positive

FP = False Positive

Recall

Recall answers:

Of all actual examples of a class, how many did the model correctly identify?

[
Recall =
\frac{TP}{TP+FN}
]

where:

FN = False Negative

F1 Score

F1 combines precision and recall.

[
F1 =
2\frac{Precision\times Recall}
{Precision+Recall}
]

It is useful when both false positives and false negatives matter.

Classification Report

Syntax:

classification_report(
    y_true,
    y_pred,
    target_names=le.classes_
)

It reports:

precision

recall

F1-score

support

for each class.

21. Confusion Matrix

A confusion matrix shows actual classes versus predicted classes.

cm = confusion_matrix(
    y_true_labels,
    y_pred_labels,
    labels=unique_tags
)

Conceptual example:

                 Predicted
               A     B     C

Actual A      10     1     0
Actual B       2     8     1
Actual C       0     1     9

The diagonal contains correct predictions.

Off-diagonal values represent misclassifications.

The project visualizes it using Seaborn:

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    ...
)

22. Hybrid Intent Detection

The NLP chatbot combines several techniques.

The flow is:

User Input
    ↓
TF-IDF preprocessing
    ↓
Cosine similarity
    ↓
Best intent
    ↓
Confidence check
    ↓
If very low:
    Rule-based regex
    ↓
Response

The code uses:

if score < 0.15:
    tag2, response = regex_match(user_input)

Otherwise the NLP prediction is used.

This is a hybrid architecture because it combines:

deterministic rules

statistical text similarity

machine learning

23. Confidence Thresholds

A similarity score can be used as a confidence-like measure.

The function:

predict_intent_cosine(user_input, threshold=0.25)

uses:

if best_score < threshold:
    best_tag = 'unknown'

So:

score >= 0.25
    ↓
accept best intent

score < 0.25
    ↓
unknown

The conversation loop additionally uses:

if score < 0.15:
    regex_match(...)

Important distinction:

0.25 is the threshold used inside predict_intent_cosine().

0.15 is used by the conversation loop to decide whether to switch to the rule-based matcher.

These thresholds are heuristic values and should be tuned using validation data rather than treated as universal NLP standards.

24. Model Serialization

Training a model every time is unnecessary.

The project uses Python's pickle module.

import pickle

Save:

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(tfidf, f)

Load:

with open(MODEL_PATH, 'rb') as f:
    tfidf = pickle.load(f)

The project stores:

chatbot_tfidf.pkl
chatbot_encoder.pkl
chatbot_labels.pkl

Why serialize models?

Without serialization:

Start program
↓
Train model
↓
Use chatbot

With serialization:

Start program
↓
Load trained model
↓
Use chatbot

This saves time.

Security warning

Never unpickle untrusted files. Python pickle can execute arbitrary code during deserialization.

25. Conversation Loop

The chatbot continuously receives user input.

Basic structure:

while True:
    user_input = input('You: ').strip()

    if not user_input:
        continue

    tag, response = regex_match(user_input)

    print(f'Bot: {response}')

    if tag == 'farewell':
        break

while True

Creates an infinite loop.

input()

Reads keyboard input.

strip()

Removes leading and trailing whitespace.

continue

Skips the current loop iteration.

break

Terminates the loop.

26. Python Libraries

Standard Library

random

Used to select a random response.

import random

random.choice(items)

Example:

response = random.choice(entry['responses'])

re

Regular expressions.

import re

re.search(pattern, text)

string

Provides punctuation constants.

import string

string.punctuation

os

File/path operations.

import os

os.path.exists(path)

pickle

Model serialization.

import pickle

NLTK

Natural Language Toolkit.

Used for:

tokenization

stop words

WordNet

linguistic preprocessing

Imports used:

import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

Resources downloaded:

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

NumPy

import numpy as np

NumPy provides numerical arrays and mathematical operations.

The current NLP script imports NumPy, although the primary intent-matching pipeline is implemented using scikit-learn's sparse matrices.

Matplotlib

import matplotlib.pyplot as plt

Used to create/save plots.

The project sets:

matplotlib.use('Agg')

This is a non-interactive backend, useful when running without a graphical display.

Seaborn

import seaborn as sns

Used to create a readable heatmap for the confusion matrix.

Example:

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

Scikit-learn

Scikit-learn provides most of the machine-learning components.

Used modules:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

FPDF

generate_report.py uses:

from fpdf import FPDF

It generates a PDF report programmatically.

27. Library Syntax Reference

TfidfVectorizer

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    analyzer='word'
)

Fit and transform:

X = vectorizer.fit_transform(documents)

Transform new text:

X_new = vectorizer.transform([new_text])

Vocabulary:

vectorizer.vocabulary_

Cosine Similarity

scores = cosine_similarity(vector_a, vector_matrix)

LabelEncoder

encoder = LabelEncoder()

encoded = encoder.fit_transform(labels)

original = encoder.inverse_transform(encoded)

Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Logistic Regression

model = LogisticRegression(
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

Probability:

probabilities = model.predict_proba(X_test)

Accuracy

accuracy = accuracy_score(y_test, predictions)

Classification Report

report = classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_
)

Confusion Matrix

cm = confusion_matrix(
    y_test,
    predictions
)

28. Installation

Requirements

Recommended:

Python 3.10+
pip
virtual environment

Create a virtual environment:

macOS/Linux

python3 -m venv venv
source venv/bin/activate

Windows

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install numpy matplotlib seaborn nltk scikit-learn fpdf2

If the project later includes a requirements.txt:

pip install -r requirements.txt

29. Running the Project

Move into the project directory:

cd MFS-Task1-main

Run Rule-Based Chatbot

python chatbot.py

Example:

CollegeBot — Your AI College Assistant

You: hello
Bot: Hello! Welcome to CollegeBot. How can I help you today?

You: what is the fee structure
Bot: Fee structure varies by course...

You: bye
Bot: Goodbye! Best of luck with your studies!

Run NLP Chatbot

python nlp_chatbot.py

If serialized model files do not exist:

No pre-trained model files found. Training now...

The program trains the model and saves the artifacts.

Later executions load the model.

30. Training Pipeline

The complete training process is:

Step 1 — Build Dataset

The system reads every pattern from the knowledge base.

for entry in KNOWLEDGE_BASE:
    for pattern in entry['patterns']:
        X_raw.append(pattern)
        X_proc.append(advanced_preprocess(pattern))
        y.append(entry['tag'])

So:

X_raw  → original sentences
X_proc → processed sentences
y      → intent labels

Step 2 — TF-IDF

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    analyzer='word'
)

X_tfidf = tfidf.fit_transform(X_proc)

Step 3 — Encode Labels

le = LabelEncoder()
y_encoded = le.fit_transform(y)

Step 4 — Split Dataset

train_test_split(
    X_tfidf,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

Step 5 — Train Classifier

lr_clf.fit(X_tr, y_tr)

Step 6 — Predict

lr_preds = lr_clf.predict(X_te)

Step 7 — Evaluate

accuracy_score(y_te, lr_preds)

and:

classification_report(
    y_te,
    lr_preds,
    target_names=le.classes_
)

Step 8 — Confusion Matrix

cm = confusion_matrix(
    y_true_labels,
    y_pred_labels,
    labels=unique_tags
)

Step 9 — Save Models

save_model()

31. Verification

verify_nlp.py tests unseen phrasings.

Example inputs:

test_inputs = [
    'hi there how are you',
    'what is the scholarship amount',
    'I need information about accommodation',
    'what jobs will I get after graduating',
    'I want to study computer science'
]

The important idea is that these are not necessarily exact copies of the training patterns.

The program calculates:

Input
↓
Preprocessing
↓
TF-IDF
↓
Cosine similarity
↓
Best matching training pattern
↓
Intent

This tests generalization to new wording.

32. Important Python Syntax

List

patterns = [
    'hello',
    'hi',
    'hey'
]

Dictionary

entry = {
    'tag': 'greeting',
    'patterns': ['hello', 'hi'],
    'responses': ['Hello!']
}

Access:

entry['tag']

Set

words = set(text.split())

Intersection:

A & B

Union:

A | B

Difference:

A - B

Function

def greet(name):
    return f"Hello {name}"

Conditional

if score > threshold:
    print("Match")
else:
    print("No match")

Loop

for item in items:
    print(item)

List Comprehension

Used in the project:

tokens = [t for t in tokens if t.isalpha()]

Equivalent longer form:

filtered = []

for t in tokens:
    if t.isalpha():
        filtered.append(t)

tokens = filtered

f-string

print(f"Score: {score:.3f}")

:.3f means three decimal places.

File Handling

Write:

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

Read:

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

Modes:

r  → read text
w  → write text
a  → append
rb → read binary
wb → write binary

33. Mathematical Theory

Set Intersection

For:

[
A={a,b,c}
]

and:

[
B={b,c,d}
]

then:

[
A\cap B={b,c}
]

The chatbot uses this idea to calculate keyword overlap.

Vector Dot Product

For:

[
A=(a_1,a_2,\ldots,a_n)
]

and:

[
B=(b_1,b_2,\ldots,b_n)
]

the dot product is:

[
A\cdot B =
\sum_{i=1}^{n}a_i b_i
]

Vector Magnitude

[
|A| =
\sqrt{\sum_{i=1}^{n}a_i^2}
]

Cosine Similarity

[
cos(\theta)=
\frac{\sum_i A_iB_i}
{\sqrt{\sum_i A_i^2}
\sqrt{\sum_i B_i^2}}
]

Sigmoid Function

Logistic Regression uses the sigmoid function:

[
\sigma(z)=\frac{1}{1+e^{-z}}
]

It maps a real number into:

0 to 1

Classification

The classifier learns a relationship:

[
f(X)\rightarrow y
]

where:

X = text features
y = intent

34. End-to-End Data Flow

Training

Knowledge Base
     ↓
Patterns + Tags
     ↓
Advanced Preprocessing
     ↓
Processed Text
     ↓
TF-IDF Vectorization
     ↓
Numeric Feature Matrix
     ↓
Label Encoding
     ↓
Train/Test Split
     ↓
Logistic Regression
     ↓
Evaluation
     ↓
Pickle Serialization

Prediction

User Input
     ↓
Lowercase
     ↓
Tokenize
     ↓
Remove punctuation/non-alpha tokens
     ↓
Remove stop words
     ↓
Lemmatize
     ↓
TF-IDF transform
     ↓
Cosine similarity
     ↓
Best training pattern
     ↓
Similarity score
     ↓
Intent
     ↓
Random response

Fallback

NLP score
    ↓
Very low?
   / \
 Yes  No
  ↓    ↓
Regex  NLP result
  ↓
Keyword overlap if regex fails

35. Rule-Based vs NLP Comparison

Feature

Rule-Based

NLP-Enhanced

Pattern matching

Yes

Yes

Regex

Yes

Fallback

Tokenization

Basic splitting

NLTK

Stop-word removal

No

Yes

Lemmatization

No

Yes

TF-IDF

No

Yes

Cosine similarity

No

Yes

ML classifier

No

Logistic Regression

Handles paraphrases

Limited

Better

Training required

No

Yes

Dataset requirement

Small rules

Labeled patterns

Interpretability

Very high

High

Flexibility

Low

Higher

Computational cost

Very low

Higher

36. Advantages

Rule-Based Advantages

simple implementation

easy to understand

deterministic

fast

no training required

easy to debug

NLP Advantages

handles more natural wording

reduces dependence on exact phrases

supports morphological normalization

captures useful word combinations using n-grams

provides similarity scores

supports supervised classification

can be expanded with more training data

Hybrid Advantages

The hybrid system combines the strengths of both approaches:

ML/NLP flexibility
        +
Rule-based reliability
        =
Hybrid chatbot

37. Limitations

Dataset Size

The knowledge base is small.

A small dataset can cause:

overfitting

unstable evaluation

poor handling of unseen intents

weak class coverage

Adding more diverse examples improves the model.

TF-IDF Limitation

TF-IDF does not truly understand language semantics.

For example:

"car"

and:

"automobile"

are treated as different tokens unless the training data provides a relationship through shared features.

Modern embedding models handle semantic relationships much better.

Cosine Similarity Limitation

Cosine similarity measures vector similarity, not deep language meaning.

Intent Ambiguity

A query may contain words belonging to multiple intents.

Example:

"What is the fee for the hostel?"

contains:

fee
hostel

A simple keyword or similarity system may choose the wrong intent depending on the training examples.

Hard-Coded Responses

Responses are predefined.

The chatbot does not dynamically generate factual answers.

38. Improvements

1. Increase Training Data

Add many variations:

"What are the fees?"
"How much do I have to pay?"
"What is the tuition cost?"
"Tell me the semester charges."

2. Add More Intents

Examples:

library
exam
scholarship
transport
faculty
contact
location
timings
departments
documents

3. Use Word Embeddings

Possible technologies:

Word2Vec

GloVe

FastText

Sentence Transformers

Embeddings represent words/sentences in dense vector spaces.

4. Use Transformer Models

Modern systems can use:

BERT

DistilBERT

RoBERTa

sentence-transformers

These generally provide much stronger semantic understanding than TF-IDF.

5. Add Confidence Calibration

Instead of treating cosine similarity directly as probability, use a validation set to choose thresholds.

6. Add Entity Extraction

Example:

"What is the fee for B.Tech CSE?"

Possible entities:

Course = B.Tech
Department = CSE

7. Add Conversation Context

Current system mostly processes each query independently.

A context-aware system could understand:

User: What is the fee for B.Tech?
Bot: ...
User: What about hostel?

The second question should be interpreted using the previous context.

39. Common Errors

NLTK Resource Error

Example:

LookupError: Resource punkt not found

Solution:

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

ModuleNotFoundError

Example:

ModuleNotFoundError: No module named 'sklearn'

Install:

pip install scikit-learn

For NLTK:

pip install nltk

For plotting:

pip install matplotlib seaborn

Model Files Missing

If:

chatbot_tfidf.pkl
chatbot_encoder.pkl
chatbot_labels.pkl

do not exist, nlp_chatbot.py automatically trains the model.

Class Split Error

With very small datasets, stratified splitting can fail if a class has too few examples.

Solution:

add more examples per class

use a larger dataset

adjust the split strategy

Poor Intent Prediction

Possible causes:

too few patterns

overlapping intents

poor preprocessing

unsuitable threshold

ambiguous queries

vocabulary mismatch

Solutions:

add training examples

add synonyms/paraphrases

tune thresholds

use embeddings

merge or redesign overlapping intents

40. Viva/Interview Questions

Q1. What is NLP?

NLP is Natural Language Processing, a field of AI that enables computers to process and analyze human language.

Q2. What is an intent?

An intent represents the purpose behind a user's message.

Example:

"What is the fee?"

Intent:

fees

Q3. Why is preprocessing required?

Raw language contains:

punctuation

capitalization differences

common words

morphological variations

Preprocessing reduces irrelevant variation and noise.

Q4. What is tokenization?

Breaking text into smaller units called tokens.

Q5. What are stop words?

Common words that often provide limited information for a particular classification task.

Q6. What is lemmatization?

Converting words into meaningful base forms.

Q7. What is TF-IDF?

A numerical text representation that weights terms according to their frequency in a document and rarity across documents.

Q8. Why use bigrams?

Bigrams preserve two-word phrases such as:

fee structure
course fee
campus recruitment

which can contain useful intent information.

Q9. What is cosine similarity?

A measure of the angular similarity between two vectors.

Q10. Why use Logistic Regression?

It is a simple and effective baseline for classification, especially with sparse TF-IDF features.

Q11. What is LabelEncoder?

It converts categorical labels into numeric values.

Q12. Why split the dataset?

To evaluate how the trained model performs on data not used for training.

Q13. What is overfitting?

Overfitting occurs when a model learns the training data too closely and performs poorly on unseen data.

Q14. What is a confusion matrix?

A table showing actual versus predicted classes.

Q15. What is precision?

The proportion of predicted positives that are actually positive.

Q16. What is recall?

The proportion of actual positives that are correctly detected.

Q17. What is F1-score?

The harmonic mean of precision and recall.

Q18. Why use pickle?

To save and reload trained Python objects such as vectorizers and classifiers.

Q19. What is a hybrid chatbot?

A chatbot that combines multiple approaches, such as ML/NLP and deterministic rules.

Q20. What is the main limitation of TF-IDF?

TF-IDF captures lexical/statistical information but does not deeply understand semantic meaning.



Knowledge Base
→ preprocessing
→ keyword overlap
→ regex
→ response

The second system introduces machine-learning concepts:

Knowledge Base
→ tokenization
→ stop-word removal
→ lemmatization
→ TF-IDF
→ cosine similarity
→ intent detection

It also includes a supervised classification pipeline:

TF-IDF
→ Label Encoding
→ Train/Test Split
→ Logistic Regression
→ Evaluation
→ Confusion Matrix

Finally, the hybrid fallback mechanism combines deterministic rules with NLP similarity to make the chatbot more robust.

The project therefore provides a practical introduction to:

Artificial Intelligence

Chatbots

Natural Language Processing

Text preprocessing

Regular expressions

Feature engineering

TF-IDF

N-grams

Vector similarity

Supervised machine learning

Logistic Regression

Model evaluation

Model persistence

Hybrid AI architectures

For a production-grade chatbot, the next major step would be to replace or complement TF-IDF with sentence embeddings or transformer-based models, expand the training dataset, introduce entity extraction and conversation memory, and connect the response layer to a verified college information database or API.
### Kaustav Mani Deka 
