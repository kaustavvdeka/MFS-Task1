import nlp_chatbot

def verify():
    print("Loading pre-trained model files...")
    if not nlp_chatbot.load_model():
        print("Model files not found! Training model first...")
        nlp_chatbot.train_model()
    
    test_inputs = [
        'hi there how are you',
        'what is the scholarship amount',  # not in patterns
        'I need information about accommodation',  # synonym for hostel
        'what jobs will I get after graduating',  # synonym for placements
        'I want to study computer science'  # synonym for courses
    ]
    
    print("\n" + "="*70)
    print(" RUNNING VERIFICATION ON NOVEL PHRASINGS")
    print("="*70)
    
    for inp in test_inputs:
        tag, resp, score = nlp_chatbot.predict_intent_cosine(inp)
        print(f"Input            : '{inp}'")
        print(f"Detected Intent  : {tag} (Cosine Similarity: {score:.3f})")
        print(f"Response Snippet : {resp[:120]}...")
        print("-" * 70)

if __name__ == '__main__':
    verify()
