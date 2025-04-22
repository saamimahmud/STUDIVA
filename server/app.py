from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import tempfile
from tempfile import NamedTemporaryFile
from datetime import datetime
import os
import json
import random
import nltk
import textstat
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import language_tool_python
import atexit # Import atexit

app = Flask(__name__)
CORS(app)

# Download NLTK data (if not already downloaded)
def download_nltk_resource(resource_id, resource_path):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        print(f"Downloading NLTK resource: {resource_id}")
        nltk.download(resource_id)

download_nltk_resource('punkt', 'tokenizers/punkt')
download_nltk_resource('stopwords', 'corpora/stopwords')
download_nltk_resource('punkt_tab', 'tokenizers/punkt_tab/english/') # Added download for punkt_tab
download_nltk_resource('vader_lexicon', 'sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt') # Add VADER download

# Initialize NLTK tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
# Initialize VADER Analyzer (can be done once at startup)
sia = SentimentIntensityAnalyzer()

# Load models and questions once during startup
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
whisper_model = whisper.load_model("base")

# Initialize LanguageTool globally
lang_tool = None
try:
    print("Initializing LanguageTool...")
    lang_tool = language_tool_python.LanguageTool('en-US')
    print("LanguageTool Initialized.")
except Exception as e:
    print(f"CRITICAL: Failed to initialize LanguageTool: {e}")
    # Optionally, exit or disable communication evaluation if LT fails

# Register cleanup function for LanguageTool
@atexit.register
def cleanup_lang_tool():
    if lang_tool:
        print("Closing LanguageTool resources...")
        try:
            lang_tool.close()
            print("LanguageTool closed.")
        except Exception as e:
            print(f"Error closing LanguageTool: {e}")

# Load questions from JSON file
try:
    with open('questions.json', 'r') as f:
        all_questions = json.load(f)
except FileNotFoundError:
    all_questions = {} # Initialize empty if file not found

# --- Helper function for improved keyword matching ---
def calculate_keyword_score(student_answer, expected_keywords):
    if not expected_keywords:
        return 100.0 # Or 0.0, depending on desired behavior

    # Tokenize and stem student answer (remove stopwords)
    answer_tokens = word_tokenize(student_answer.lower())
    answer_stems = {stemmer.stem(token) for token in answer_tokens if token.isalnum() and token not in stop_words}

    found_keywords_count = 0
    for keyword_phrase in expected_keywords:
        # Tokenize and stem the keyword phrase
        keyword_tokens = word_tokenize(keyword_phrase.lower())
        keyword_stems = {stemmer.stem(token) for token in keyword_tokens if token.isalnum()} # Keep all keyword tokens

        # Check if all stems of the keyword phrase are in the answer stems
        if keyword_stems.issubset(answer_stems):
            found_keywords_count += 1

    return round((found_keywords_count / len(expected_keywords)) * 100, 2)
# --- End Helper Function ---

@app.route('/questions', methods=['GET'])
def get_questions():
    subject = request.args.get('subject')
    mode = request.args.get('mode', 'sequential') # Default to sequential

    if not subject or subject not in all_questions:
        return jsonify({"error": "Subject not found or not specified"}), 404

    questions_for_subject = all_questions[subject]

    # Handle different modes (placeholder for custom/uploaded logic)
    if mode == 'random':
        random.shuffle(questions_for_subject)
    elif mode == 'custom':
        # Placeholder: In future, filter for custom questions or fetch from DB
        pass
    elif mode == 'uploaded':
        # Placeholder: In future, fetch teacher-uploaded questions
        pass
    # Default is sequential (as loaded from file)

    return jsonify(questions_for_subject)


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    student_answer = data.get("answer", "")
    expected_answer = data.get("expected", "")
    expected_keywords = data.get("keywords", [])

    if not student_answer or not expected_answer:
        return jsonify({"error": "Missing answer or expected response"}), 400

    # 1. Semantic Similarity (existing logic)
    embeddings = model.encode([student_answer, expected_answer])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    similarity_score = round(similarity * 100, 2)

    # 2. Improved Keyword Matching (using helper function)
    keyword_score = calculate_keyword_score(student_answer, expected_keywords)

    # 3. Fluency/Coherence Score (Flesch Reading Ease)
    try:
        fluency_score = textstat.flesch_reading_ease(student_answer)
    except Exception:
         # Handle cases with very short text or errors during calculation
        fluency_score = 0 # Assign a default/neutral score

    # --- Feedback Generation (can be further refined based on new scores) ---
    # Existing logic based on similarity_score remains for now
    if similarity_score > 85:
        feedback = "Excellent! Your answer closely matches the expected response and uses relevant keywords."
        confidence = "High"
    elif similarity_score > 60:
        feedback = "Good attempt. You covered the main idea. Check if you used all the key terms."
        confidence = "Medium"
    elif similarity_score > 40:
        feedback = "Some relevant parts, but the answer needs improvement in clarity and keyword usage."
        confidence = "Low"
    else:
        feedback = "The answer is mostly incorrect or irrelevant. Please review the topic."
        confidence = "Very Low"
    # You could add more nuanced feedback based on combinations of similarity, keyword, and fluency scores here.
    # Example: if similarity_score > 60 and keyword_score < 50: feedback += " Try to incorporate more keywords."
    # Example: if fluency_score < 50: feedback += " Consider rephrasing for better clarity."
    # --- End Feedback Generation ---

    return jsonify({
        "similarity_score": similarity_score,
        "keyword_score": keyword_score,
        "fluency_score": fluency_score, # Add fluency score
        "feedback": feedback,
        "confidence": confidence
    })


# Simple keyword mapping for scenarios (expand as needed)
scenario_keywords = {
    "Ask a doubt": ["doubt", "question", "understand", "clarify", "explain"],
    "Explain why you came late": ["late", "sorry", "apologize", "reason", "traffic", "overslept"],
    "Politely disagree": ["disagree", "point", "however", "alternative", "perspective", "but"],
    "Respond to constructive feedback": ["feedback", "understand", "improve", "suggestion", "point", "thank"],
    "Ask the teacher for an extension": ["extension", "deadline", "time", "assignment", "submit", "request"]
}

@app.route('/evaluate_communication', methods=['POST'])
def evaluate_communication():
    data = request.get_json()
    scenario = data.get("scenario", "")
    response_text = data.get("response", "")

    if not scenario or not response_text:
        return jsonify({"error": "Missing scenario or response text"}), 400

    # Check if LanguageTool was initialized successfully
    if lang_tool is None:
        return jsonify({"error": "LanguageTool failed to initialize. Grammar check unavailable."}), 503 # Service Unavailable

    try:
        # Use the global lang_tool instance
        # Remove: lang_tool = language_tool_python.LanguageTool('en-US') 

        # 1. Grammar Checking
        grammar_matches = lang_tool.check(response_text)
        num_grammar_errors = len(grammar_matches)
        if num_grammar_errors == 0:
            grammar_score = "Good"
        elif num_grammar_errors <= 2:
            grammar_score = "Minor issues"
        else:
            grammar_score = "Check grammar"
        grammar_feedback_examples = [match.message for match in grammar_matches[:2]]

        # 2. Politeness & Tone (Using VADER)
        sentiment_scores = sia.polarity_scores(response_text)
        compound_score = sentiment_scores['compound']
        positive_score = sentiment_scores['pos']
        negative_score = sentiment_scores['neg']

        # Politeness Score
        if compound_score >= 0.5:
            politeness_score = "Good (Positive Sentiment)"
        elif compound_score > -0.1: # Allow slightly negative for neutral
            politeness_score = "Okay (Neutral/Slightly Positive Sentiment)"
        else:
            politeness_score = "Needs Improvement (Negative Sentiment)"
        politeness_sentiment_value = f"(Sentiment Score: {compound_score:.2f})"
        
        # Tone Score
        tone = "Neutral"
        if positive_score > negative_score and positive_score > 0.2: # Threshold for clearly positive
            tone = "Positive"
        elif negative_score > positive_score and negative_score > 0.2: # Threshold for clearly negative
            tone = "Negative"
        tone_value = f"(Pos: {positive_score:.2f}, Neg: {negative_score:.2f})"

        # 3. Clarity (existing logic)
        try:
            clarity_raw_score = textstat.flesch_reading_ease(response_text)
            if clarity_raw_score >= 60:
                 clarity_score = "Clear"
            elif clarity_raw_score >= 40:
                 clarity_score = "Mostly Clear"
            else:
                 clarity_score = "Could be clearer"
        except Exception: clarity_score = "Could be clearer"

        # 4. Appropriateness (Length + Basic Keyword Check)
        response_lower = response_text.lower()
        words = response_text.split()
        word_count = len(words)
        relevant_keywords_found = False
        appropriateness_issue = ""

        # Find relevant keywords for the current scenario
        current_keywords = []
        # Match scenario description loosely (case-insensitive substring match)
        matched_key = next((key for key in scenario_keywords if key.lower() in scenario.lower()), None)
        if matched_key:
            current_keywords = scenario_keywords[matched_key]

        if current_keywords:
            for keyword in current_keywords:
                if keyword in response_lower:
                    relevant_keywords_found = True
                    break # Found at least one relevant keyword
        else: 
            # If no keywords defined for scenario, assume relevance check passes
            relevant_keywords_found = True 

        # Determine appropriateness score/feedback
        if word_count < 5:
            appropriateness_score = "Too brief"
            appropriateness_issue = "Response seems very short for the scenario."
        elif word_count > 100:
            appropriateness_score = "Too long"
            appropriateness_issue = "Response seems quite long for the scenario."
        elif not relevant_keywords_found and current_keywords:
            appropriateness_score = "Potentially Off-Topic"
            appropriateness_issue = f"Response doesn\'t seem to contain keywords related to the scenario (e.g., {', '.join(current_keywords[:3])}). Ensure you address the scenario directly."
        else:
            appropriateness_score = "Okay"
            appropriateness_issue = "Length and basic relevance seem appropriate."

        # --- Feedback Generation --- 
        feedback_parts = []
        feedback_parts.append(f"Grammar: {grammar_score}.")
        if grammar_feedback_examples:
             feedback_parts.append(f" Examples: {', '.join(grammar_feedback_examples)}")

        feedback_parts.append(f"\nTone: {tone} {tone_value}.") # Added Tone
        feedback_parts.append(f"\nPoliteness: {politeness_score} {politeness_sentiment_value}.")
        if compound_score <= 0:
             feedback_parts.append(" Try using more positive phrasing or explicitly polite words if appropriate.")

        feedback_parts.append(f"\nClarity (Readability): {clarity_score}.")
        if clarity_score == "Could be clearer":
             feedback_parts.append(" Try using simpler sentences.")

        feedback_parts.append(f"\nAppropriateness: {appropriateness_score}.") # Updated Appropriateness score text
        feedback_parts.append(f" {appropriateness_issue}") # Add the detailed issue

        final_feedback = "".join(feedback_parts) # Join without extra space, use newlines in f-strings

        # Remove lang_tool.close() from here

        return jsonify({
            "politeness": politeness_score,
            "clarity": clarity_score,
            "grammar": grammar_score,
            "appropriateness": appropriateness_score,
            "tone": tone, # Added Tone to response
            "feedback": final_feedback
        })

    except Exception as e:
        # Error handling no longer needs to close lang_tool here
        print(f"Error during communication evaluation: {e}")
        return jsonify({"error": "An internal error occurred during evaluation."}), 500

@app.route('/save_session', methods=['POST'])
def save_session():
    session_data = request.get_json()

    # Create folder if not exists
    os.makedirs('saved_sessions', exist_ok=True)

    # Use timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"saved_sessions/session_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(session_data, f, indent=4)

    return jsonify({"message": "Session saved successfully", "filename": filename})



@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files['audio']

    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as temp:
            temp.write(audio_file.read())
            temp.flush()
            temp_path = temp.name

        result = whisper_model.transcribe(temp_path)
        transcript = result["text"]
        return jsonify({"transcript": transcript})

    except Exception as e:
        return jsonify({"error": f"Failed to transcribe audio: {str(e)}"}), 500
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                pass  # File still in use — skip cleanup

# --- Helper Function to Save Questions --- 
QUESTION_FILE = 'questions.json'

def save_all_questions():
    """Saves the current state of all_questions to the JSON file."""
    try:
        with open(QUESTION_FILE, 'w') as f:
            json.dump(all_questions, f, indent=2) # Use indent=2 for readability
        return True
    except Exception as e:
        print(f"ERROR saving questions to {QUESTION_FILE}: {e}")
        return False

# --- Admin API Endpoints --- 

@app.route('/admin/questions', methods=['GET'])
def admin_get_all_questions():
    """Returns the entire question bank."""
    # Return a copy to avoid potential modification issues if needed later
    return jsonify(dict(all_questions))

@app.route('/admin/questions/<subject>', methods=['POST'])
def admin_add_question(subject):
    """Adds a new question to a subject."""
    data = request.get_json()
    if not data or 'question' not in data or 'expected' not in data:
        return jsonify({"error": "Missing question data (question, expected fields required)"}), 400

    new_question = {
        "question": data['question'],
        "expected": data['expected'],
        "keywords": data.get('keywords', []) # Keywords are optional
    }

    if subject not in all_questions:
        all_questions[subject] = [] # Create subject if it doesn't exist
        
    all_questions[subject].append(new_question)
    
    if save_all_questions():
        # Return the added question and its new index
        return jsonify({"message": "Question added successfully", "question": new_question, "index": len(all_questions[subject]) - 1}), 201
    else:
        # Rollback the change in memory if save failed
        all_questions[subject].pop()
        if not all_questions[subject]: # Remove subject if it became empty
            del all_questions[subject]
        return jsonify({"error": "Failed to save questions after adding."}), 500

@app.route('/admin/questions/<subject>/<int:question_index>', methods=['PUT'])
def admin_update_question(subject, question_index):
    """Updates an existing question."""
    if subject not in all_questions or question_index >= len(all_questions[subject]):
        return jsonify({"error": "Subject or question index not found"}), 404

    data = request.get_json()
    if not data or 'question' not in data or 'expected' not in data:
        return jsonify({"error": "Missing question data (question, expected fields required)"}), 400
        
    # Keep the original question in case we need to rollback
    original_question = dict(all_questions[subject][question_index]) 

    # Update the question in memory
    all_questions[subject][question_index] = {
        "question": data['question'],
        "expected": data['expected'],
        "keywords": data.get('keywords', original_question.get('keywords', [])) # Keep original keywords if not provided
    }

    if save_all_questions():
        return jsonify({"message": "Question updated successfully", "question": all_questions[subject][question_index]}), 200
    else:
        # Rollback on save failure
        all_questions[subject][question_index] = original_question
        return jsonify({"error": "Failed to save questions after update."}), 500

@app.route('/admin/questions/<subject>/<int:question_index>', methods=['DELETE'])
def admin_delete_question(subject, question_index):
    """Deletes a question."""
    if subject not in all_questions or question_index >= len(all_questions[subject]):
        return jsonify({"error": "Subject or question index not found"}), 404
        
    # Store the question to potentially add back if save fails
    deleted_question = all_questions[subject].pop(question_index)
    subject_was_emptied = not all_questions[subject]
    if subject_was_emptied:
        del all_questions[subject] # Remove subject key if empty

    if save_all_questions():
        return jsonify({"message": "Question deleted successfully"}), 200
    else:
        # Rollback on save failure
        if subject_was_emptied:
             all_questions[subject] = [deleted_question] # Recreate subject list
        else:
            all_questions[subject].insert(question_index, deleted_question) # Put question back
        return jsonify({"error": "Failed to save questions after deletion."}), 500

if __name__ == '__main__':
    app.run(debug=True)
