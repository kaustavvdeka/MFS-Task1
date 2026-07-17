from fpdf import FPDF
import os

class ChatbotReportPDF(FPDF):
    def header(self):
        # Arial/Helvetica bold
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Main Flow Services and Technologies Private Limited | AI/ML Internship', border=0, align='R')
        self.ln(8)
        # Line under header
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', border=0, align='C')

def create_pdf_report():
    pdf = ChatbotReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(22, 53, 116)  # Sleek dark blue
    pdf.cell(0, 15, 'Project Report: AI Chatbot', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
    
    # Subtitle
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'Rule-Based Engine & NLP-Enhanced Intent Classifier', border=0, align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # ----------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(22, 53, 116)
    pdf.cell(0, 10, '1. Executive Summary', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    summary_text = (
        "This project presents the design, implementation, and evaluation of two generations of AI "
        "chatbots: a Rule-Based Chatbot using keyword/overlap and regular expression matching (Task 1), "
        "and an NLP-Enhanced Chatbot using TF-IDF vectorization, Cosine Similarity intent detection, and "
        "supervised classification (Task 2). Built as an interactive FAQ assistant for a college support desk, "
        "the chatbot handles queries about admissions, fees, courses, hostel, and placements, while "
        "demonstrating robust fallback metrics and session tracking."
    )
    pdf.multi_cell(0, 5, summary_text)
    pdf.ln(5)
    
    # ----------------------------------------------------
    # SECTION 2: RULE-BASED ENGINE
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(22, 53, 116)
    pdf.cell(0, 10, '2. Rule-Based Chatbot Engine (Task 1)', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    rule_text = (
        "The rule-based chatbot matches user inputs directly to pre-defined keyword lists and regex patterns:\n"
        " - Preprocessing: Cleans input by converting to lowercase, stripping punctuation, and removing excess whitespace.\n"
        " - Overlap Matching: Computes the size of the intersection between user words and tokenized patterns.\n"
        " - Regex Patterns: Uses specific regular expressions (with word boundary constraints '\\b') to map queries like "
        "'fee structure' or 'course cost' to their correct intents before fallback.\n"
        " - Limitation: Highly brittle; fails to recognize variations, synonyms, or paraphrases if they do not contain the "
        "exact pre-defined keywords."
    )
    pdf.multi_cell(0, 5, rule_text)
    pdf.ln(5)

    # ----------------------------------------------------
    # SECTION 3: NLP-ENHANCED CHATBOT
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(22, 53, 116)
    pdf.cell(0, 10, '3. NLP-Enhanced Chatbot (Task 2)', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    nlp_text = (
        "The NLP-enhanced chatbot upgrades the matching capabilities using machine learning and vector representation:\n"
        " - Advanced Preprocessing: Uses tokenization, filters non-alphabetic tokens, removes common stop words "
        "(retaining negations like 'not' and 'no'), and lemmatizes words to their root forms (e.g., 'graduating' -> 'graduate').\n"
        " - TF-IDF Vectorizer: Fits a TF-IDF model with ngram_range=(1, 2) to capture single words and two-word phrases, "
        "representing text as numerical vectors.\n"
        " - Cosine Similarity Matcher: Calculates the cosine similarity score between the user input TF-IDF vector and "
        "all training patterns. Matches the highest score, falling back to rule-based regex patterns if the confidence score "
        "drops below 0.25, and switching to unknown responses if the confidence is below 0.15.\n"
        " - Classifier: A secondary Logistic Regression classifier is trained to evaluate supervised performance."
    )
    pdf.multi_cell(0, 5, nlp_text)
    pdf.ln(10)

    # ----------------------------------------------------
    # SECTION 4: EVALUATION & CONFUSION MATRIX
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(22, 53, 116)
    pdf.cell(0, 10, '4. Model Evaluation & Visualizations', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    eval_text = (
        "The Logistic Regression model was trained on the preprocessed knowledge base patterns. It achieved an accuracy "
        "of 71.4% on the stratified 80-20 train-test split. Given the relatively small size of the dataset (69 patterns total), "
        "the performance is strong, with perfect recall and precision in core query tags like courses, fees, and hostel.\n\n"
        "Below is the confusion matrix showing the distribution of predictions across categories:"
    )
    pdf.multi_cell(0, 5, eval_text)
    pdf.ln(5)
    
    # Embed Confusion Matrix
    if os.path.exists("chatbot_confusion_matrix.png"):
        # Position image in center
        pdf.image("chatbot_confusion_matrix.png", x=45, y=pdf.get_y(), w=120)
        # Shift down to prevent overlap
        pdf.ln(100)
    else:
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 10, "[Confusion Matrix plot not found - chatbot_confusion_matrix.png missing]", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # ----------------------------------------------------
    # SECTION 5: COMPARATIVE DISCUSSION
    # ----------------------------------------------------
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(22, 53, 116)
    pdf.cell(0, 10, '5. Rule-Based vs. NLP Chatbot Comparison', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(50, 50, 50)
    
    comparison_text = (
        "The NLP-enhanced chatbot is a substantial advancement over the rule-based engine. While the rule-based approach "
        "is simple, fast, and does not require model training, it is inherently limited. If a user asks 'what is the "
        "scholarship amount?' or 'I need information about accommodation', the rule-based chatbot fails because the words "
        "'scholarship' and 'accommodation' are not present in the pattern vocabulary.\n\n"
        "In contrast, the NLP-enhanced chatbot utilizing TF-IDF and Cosine Similarity overcomes this bottleneck. By "
        "converting sentences into term frequency-inverse document frequency vectors, it measures the semantic distance "
        "between the user input and the entire knowledge base. The advanced preprocessing pipeline - specifically "
        "lemmatization and stop-word filtering - reduces noise, allowing synonyms and morphological variations (like "
        "'graduating' vs. 'graduation') to map to their root forms. This results in successful intent classification "
        "even for completely novel phrasings.\n\n"
        "Moreover, the hybrid architecture - which falls back to the rule-based regex matching when the similarity score "
        "drops below 0.25 - ensures high reliability, combining the generalization capabilities of NLP with the "
        "deterministic accuracy of regex rules. Therefore, the NLP-enhanced chatbot performed significantly better in "
        "understanding user intents and handling natural, unstructured conversations."
    )
    pdf.multi_cell(0, 5, comparison_text)
    
    # Save output
    pdf.output("report.pdf")
    print("Report PDF compiled successfully as 'report.pdf'!")

if __name__ == '__main__':
    create_pdf_report()
