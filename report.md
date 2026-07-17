# Project Report: AI Chatbot with Rule-Based & NLP Reasoning
**Main Flow Services and Technologies Private Limited**
*AI/ML Internship | Project 1: AI Chatbot*

---

## Executive Summary
This project presents the design, implementation, and evaluation of two generations of AI chatbots: a Rule-Based Chatbot using keyword/overlap and regular expression matching (Task 1), and an NLP-Enhanced Chatbot using TF-IDF vectorization, Cosine Similarity intent detection, and Logistic Regression classification (Task 2). The final product is built for a college FAQ assistant, capable of answering queries regarding admission, fees, courses, hostel, and placements, with graceful unknown input fallbacks and session tracking.

---

## Task 1: Rule-Based Chatbot Engine
The rule-based chatbot matches user inputs directly to pre-defined keyword lists and regex patterns:
- **Preprocessing**: Cleans input by converting to lowercase, stripping punctuation, and removing excess whitespace.
- **Overlap Score**: Computes the size of the intersection between user words and tokenized patterns.
- **Regex Patterns**: Uses specific regular expressions (with word boundary constraints `\b`) to map phrases like "fee structure" or "course cost" to their correct intents before fallback.
- **Limitation**: Highly brittle; fails to recognize variations, synonyms, or paraphrases if they do not contain the exact pre-defined keywords.

---

## Task 2: NLP-Enhanced Chatbot
The NLP-enhanced chatbot upgrades the matching capabilities using machine learning and vector representation:
- **Advanced Preprocessing**: Uses NLTK for tokenization, filters non-alphabetic tokens, removes stop words (while retaining negations like "not" or "no"), and lemmatizes words (e.g., "graduating" -> "graduate").
- **TF-IDF Vectorizer**: Fits a TF-IDF model with `ngram_range=(1, 2)` to capture single words and two-word phrases, representing text as numerical vectors.
- **Cosine Similarity Matcher**: Calculates the cosine of the angle between the user input TF-IDF vector and all training patterns. Matches the highest score, falling back to rule-based regex patterns if the confidence score drops below `0.25`, and switching to unknown responses if the confidence is below `0.15`.
- **Logistic Regression Classifier**: A secondary supervised model trained on stratified splits (80% train, 20% test) to classify queries.

---

## Evaluation Results
The Logistic Regression classifier was trained on the preprocessed knowledge base patterns. Since the dataset comprises standard query formulations, the model achieves high accuracy on the stratified test split.

### Confusion Matrix
The confusion matrix (saved as `chatbot_confusion_matrix.png`) illustrates the distribution of correct vs. incorrect classifications across all 8 target intents: `greeting`, `farewell`, `admission`, `fees`, `courses`, `hostel`, `placements`, and `thanks`. 

---

## Comparative Analysis: Rule-Based vs. NLP Chatbot
*(Which approach worked better?)*

The NLP-enhanced chatbot is a substantial advancement over the rule-based engine. 

While the rule-based approach is simple, fast, and does not require model training, it is inherently limited. If a user asks *"what is the scholarship amount?"* or *"I need information about accommodation"*, the rule-based chatbot fails because the words "scholarship" and "accommodation" are not present in the pattern vocabulary.

In contrast, the NLP-enhanced chatbot utilizing TF-IDF and Cosine Similarity overcomes this bottleneck. By converting sentences into term frequency-inverse document frequency vectors, it measures the semantic distance between the user input and the entire knowledge base. The advanced preprocessing pipeline—specifically **lemmatization** and **stop-word filtering**—reduces noise, allowing synonyms and morphological variations (like "graduating" vs. "graduation") to map to their root forms. This results in successful intent classification even for completely novel phrasings.

Moreover, the hybrid architecture—which falls back to the rule-based regex matching when the similarity score drops below `0.25`—ensures high reliability, combining the generalization capabilities of NLP with the deterministic accuracy of regex rules. Therefore, the **NLP-enhanced chatbot performed significantly better** in understanding user intents and handling natural, unstructured conversations.

---

## Conclusion
We have successfully implemented and tested both chatbot frameworks. The final product meets all requirements, handles edge cases (punctuation, multiple spaces, negative terms, out-of-vocabulary inputs), and persists the models via serialized pickle files (`.pkl`) for efficient run-time prediction.
