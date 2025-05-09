from dotenv import load_dotenv
load_dotenv()  # <-- this will read server/.env
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
import firebase_admin # Add Firebase import
from firebase_admin import credentials, firestore, auth # Add specific imports
import uuid
from functools import wraps # For decorator
from bert_score import score as bert_score_calculate # Import BERTScore
import google.generativeai as genai # Import for Gemini API
import spacy # Import spaCy
from dotenv import load_dotenv  # Import load_dotenv to load .env file

app = Flask(__name__)
CORS(app)

# Load environment variables from .env file
load_dotenv()

# --- Initialize Firebase Admin SDK ---
try:
    # First try loading from environment variable
    firebase_service_account = os.environ.get('FIREBASE_SERVICE_ACCOUNT')
    if firebase_service_account:
        service_account_info = json.loads(firebase_service_account)
        cred = credentials.Certificate(service_account_info)
    else:
        # For local development, try to load directly from file
        cred_path = os.path.join(os.path.dirname(__file__), 'firebase-admin-key.json')
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
        else:
            # Finally, try GOOGLE_APPLICATION_CREDENTIALS
            key_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if not key_path:
                raise ValueError('Could not find Firebase credentials. Please ensure firebase-admin-key.json exists in the server directory or set GOOGLE_APPLICATION_CREDENTIALS environment variable')
            cred = credentials.Certificate(key_path)

    print('Initializing Firebase Admin SDK...')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print('Firebase Admin Initialized Successfully.')
except Exception as e:
    print(f'CRITICAL: Failed to initialize Firebase Admin SDK: {e}')
    db = None  # Set db to None if initialization fails

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
print("Loading Sentence Transformer model...")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Sentence Transformer model loaded.")

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# Load spaCy model
nlp_spacy = None
try:
    print("Loading spaCy model (en_core_web_sm)...")
    nlp_spacy = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except Exception as e:
    print(f"WARNING: Failed to load spaCy model: {e}")
    # Features relying on spaCy might be disabled

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
# QUESTION_FILE = 'questions.json'
# try:
#     with open(QUESTION_FILE, 'r') as f:
#         all_questions = json.load(f)
# except FileNotFoundError:
#     all_questions = {} # Initialize empty if file not found

# Load scenarios from JSON file
# SCENARIO_FILE = 'scenarios.json'
# try:
#     with open(SCENARIO_FILE, 'r') as f:
#         all_scenarios = json.load(f) # Load as a list
# except FileNotFoundError:
#     all_scenarios = [] # Initialize empty list if file not found

# --- Helper function for improved keyword matching (USING SPACY) ---
def calculate_keyword_score_spacy(student_answer, expected_keywords):
    if not expected_keywords:
        return 100.0
    if not nlp_spacy: # Check if spaCy model loaded
        print("WARNING: spaCy model not loaded, skipping keyword scoring.")
        return 0.0 # Or handle appropriately

    try:
        # Process student answer
        doc_answer = nlp_spacy(student_answer.lower())
        answer_lemmas = {token.lemma_ for token in doc_answer if not token.is_stop and not token.is_punct}
        
        found_keywords_count = 0
        for keyword_phrase in expected_keywords:
            # Process keyword phrase
            doc_keyword = nlp_spacy(keyword_phrase.lower())
            keyword_lemmas = {token.lemma_ for token in doc_keyword if not token.is_punct}
            
            # Check if all keyword lemmas are present in the answer lemmas
            if keyword_lemmas.issubset(answer_lemmas):
                found_keywords_count += 1
                
        return round((found_keywords_count / len(expected_keywords)) * 100, 2)
    except Exception as e:
        print(f"Error during spaCy keyword scoring: {e}")
        return 0.0 # Return 0 score on error
# --- End Helper Function (spaCy) ---

# --- Helper function for Concept Matching ---
def calculate_concept_match_score(student_answer, expected_answer):
    if not nlp_spacy or not model: # Check if models are loaded
        print("WARNING: spaCy or Sentence Transformer model not loaded, skipping concept matching.")
        return 0.0

    try:
        doc_student = nlp_spacy(student_answer)
        doc_expected = nlp_spacy(expected_answer)

        student_chunks = [chunk.text for chunk in doc_student.noun_chunks]
        expected_chunks = [chunk.text for chunk in doc_expected.noun_chunks]

        if not student_chunks or not expected_chunks:
            # If either text has no noun chunks, concept match is poor
            return 0.0 

        # Encode the noun chunks
        student_embeddings = model.encode(student_chunks)
        expected_embeddings = model.encode(expected_chunks)

        # Calculate cosine similarity between all pairs
        similarity_matrix = cosine_similarity(student_embeddings, expected_embeddings)

        # Find the best match for each student chunk in the expected chunks
        # and average these best scores
        avg_max_similarity = similarity_matrix.max(axis=1).mean()
        
        # Alternative: Could also calculate the reverse (best match for expected in student)
        # avg_max_similarity_rev = similarity_matrix.max(axis=0).mean()
        # Combine them? e.g., (avg_max_similarity + avg_max_similarity_rev) / 2

        return round(float(avg_max_similarity) * 100, 2)

    except Exception as e:
        print(f"Error during concept matching: {e}")
        return 0.0
# --- End Helper Function (Concept Match) ---

# --- Authentication Decorator --- 
def check_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header missing or invalid"}), 401

        id_token = auth_header.split('Bearer ')[1]
        try:
            # Verify the ID token while checking if the token is revoked.
            decoded_token = auth.verify_id_token(id_token)
            
            # --- AUTHORIZATION CHECK --- 
            # Check for teacher custom claim
            if not decoded_token.get('teacher') == True:
                 print(f"Authorization failed: User {decoded_token.get('uid')} is not a teacher.")
                 return jsonify({"error": "Forbidden: User does not have teacher privileges"}), 403
            
            # Add user info to request context if needed by route (optional)
            # g.user = decoded_token 
            print(f"Authorized teacher access: {decoded_token.get('uid')}")
            
        except auth.InvalidIdTokenError as e:
            print(f"Token verification failed: {e}")
            return jsonify({"error": "Invalid authentication token"}), 401
        except Exception as e:
            print(f"Error during token verification: {e}")
            return jsonify({"error": "Authentication error"}), 500

        return f(*args, **kwargs)
    return decorated_function

# === Viva Questions / Subjects Routes (Public) ===

@app.route('/subjects', methods=['GET'])
def get_subjects():
    """Returns a list of available subject names from Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        subjects_ref = db.collection('vivaQuestions').stream()
        subjects = set() # Use a set to automatically handle duplicates
        for doc in subjects_ref:
            data = doc.to_dict()
            if data and 'subject' in data:
                subjects.add(data['subject'])
        
        sorted_subjects = sorted(list(subjects))
        print(f"Returning subjects from Firestore: {sorted_subjects}") 
        return jsonify(sorted_subjects)
    except Exception as e:
        print(f"Error fetching subjects from Firestore: {e}")
        return jsonify({"error": "Failed to retrieve subjects from database"}), 500

@app.route('/questions', methods=['GET'])
def get_questions():
    """Fetches questions for a given subject and mode from Firestore."""
    subject_name = request.args.get('subject')
    mode = request.args.get('mode', 'sequential') # Default to sequential

    if not subject_name:
        return jsonify({"error": "Subject parameter is required"}), 400
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        # Query Firestore for the document with the matching subject name
        # Assumes 'subject' field exists at the top level of documents in 'vivaQuestions'
        query = db.collection('vivaQuestions').where('subject', '==', subject_name).limit(1).stream()
        
        target_doc_snapshot = None
        for doc in query:
            target_doc_snapshot = doc # Get the first document found
            break # We only expect one document per subject name
            
        if not target_doc_snapshot or not target_doc_snapshot.exists:
            print(f"Subject '{subject_name}' not found in Firestore.")
            # Return empty list, as frontend expects an array even if subject not found
            return jsonify([]), 200 
        
        subject_data = target_doc_snapshot.to_dict()
        # Get the 'questions' array from the document
        questions = subject_data.get('questions', [])

        if not questions:
             print(f"No questions found within the document for subject '{subject_name}'.")
             return jsonify([]) # Return empty list if no questions found for subject

        # Handle mode (random or sequential)
        if mode == 'random':
            random.shuffle(questions)
            # Optionally limit the number of random questions:
            # questions = questions[:10] 

        # Sequential is the default; Firestore retrieval order isn't strictly guaranteed,
        # but for sequential mode, we return them as they are in the array.

        print(f"Returning {len(questions)} questions for subject '{subject_name}' (mode: {mode}).")
        return jsonify(questions)

    except Exception as e:
        print(f"Error fetching questions for subject '{subject_name}' from Firestore: {e}")
        return jsonify({"error": "Failed to retrieve questions from database"}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    student_answer = data.get("answer", "")
    expected_answer = data.get("expected", "")
    expected_keywords = data.get("keywords", [])

    if not student_answer or not expected_answer:
        return jsonify({"error": "Missing answer or expected response"}), 400

    # 1. Semantic Similarity (existing - Cosine)
    embeddings = model.encode([student_answer, expected_answer])
    cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    similarity_score = round(cosine_sim * 100, 2)

    # 2. BERTScore (New)
    try:
        # Calculate BERTScore
        # lang="en" is default, specify if needed. Using default model.
        # verbose=True prints progress.
        P, R, F1 = bert_score_calculate([student_answer], [expected_answer], lang="en", model_type="distilroberta-base", verbose=False)
        # F1 is a tensor, get the float value
        bert_f1_score = round(F1.item() * 100, 2) 
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        bert_f1_score = 0 # Default to 0 on error

    # 3. Keyword Matching (using SPAcY helper function)
    keyword_score = calculate_keyword_score_spacy(student_answer, expected_keywords)

    # 4. Concept Matching (New)
    concept_match_score = calculate_concept_match_score(student_answer, expected_answer)

    # 5. Fluency/Coherence Score (Flesch Reading Ease)
    try:
        fluency_score = textstat.flesch_reading_ease(student_answer)
    except Exception:
        fluency_score = 0 
        
    # 6. LLM Evaluation (Now using Gemini API)
    llm_evaluation_text = "Not available"
    llm_confidence = "N/A"
    llm_reasoning = "No reasoning provided." 

    try:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            print("CRITICAL: GEMINI_API_KEY environment variable not set.")
            raise ValueError("GEMINI_API_KEY not set.")
        
        genai.configure(api_key=gemini_api_key)
        
        # Model choice: Use the specific model for your API key
        gemini_model = genai.GenerativeModel(model_name='gemini-2.0-flash')

        # --- Define a prompt for Gemini (similar structure to before) ---
        prompt = f"""Your task is to evaluate the student's answer based on the expected answer.
Your MUST follow this format strictly:
1. Provide a detailed reasoning for your evaluation.
2. State if the student's answer is Correct, Partially Correct, or Incorrect.
3. Give a confidence level (High, Medium, Low).

Expected Answer: "{expected_answer}"
Student Answer: "{student_answer}"

Output ONLY in the following format:
Reasoning:
[Your detailed reasoning here. This section is mandatory and must not be empty.]

Evaluation:
[Correct/Partially Correct/Incorrect]

Confidence:
[High/Medium/Low]
"""
        # ---------------------------------------------

        # Generate content using Gemini
        # Note: For more control, explore generation_config options in Gemini SDK
        response = gemini_model.generate_content(prompt)
        llm_output_text = response.text # Accessing the text directly
        
        # --- DEBUG: Print the raw output from the LLM --- 
        print("-" * 20) # Separator
        print(f"DEBUG: Raw Gemini Output:\n{llm_output_text}")
        print("-" * 20) # Separator
        # --------------------------------------------------

        # --- Parse Reasoning, Evaluation, and Confidence (Existing Robust Logic) --- 
        reasoning_part = llm_output_text # Default to full output
        evaluation_part = "Not specified"
        confidence_part = "Not specified"

        try:
            # Use lower case for case-insensitive matching
            output_lower = llm_output_text.lower()
            reasoning_marker_lower = "reasoning:"
            eval_marker_lower = "\nfinal evaluation:" # Look for newline
            alt_eval_marker_lower = "evaluation:" # Alternative marker
            conf_marker_lower = "\nconfidence:" # Look for newline

            eval_marker_pos = output_lower.find(eval_marker_lower)
            # If standard marker not found, try alternative
            if eval_marker_pos == -1:
                eval_marker_pos = output_lower.find(alt_eval_marker_lower)
                # Adjust marker length if alternative was found
                eval_marker_len = len(alt_eval_marker_lower) if eval_marker_pos != -1 else 0
            else:
                eval_marker_len = len(eval_marker_lower)

            conf_marker_pos = output_lower.find(conf_marker_lower)
            reasoning_marker_pos = output_lower.find(reasoning_marker_lower)

            # --- Extract Parts based on found markers ---
            if eval_marker_pos != -1:
                # Extract Reasoning (if marker exists before evaluation)
                if reasoning_marker_pos != -1 and reasoning_marker_pos < eval_marker_pos:
                        reasoning_part = llm_output_text[reasoning_marker_pos + len(reasoning_marker_lower):eval_marker_pos].strip()
                else: # Otherwise, take everything before evaluation as reasoning
                        reasoning_part = llm_output_text[:eval_marker_pos].strip()
                
                # Extract Evaluation and Confidence
                evaluation_and_conf = llm_output_text[eval_marker_pos + eval_marker_len:].strip()
                
                if conf_marker_pos != -1 and conf_marker_pos > eval_marker_pos: # Ensure confidence marker is *after* eval marker
                    # Calculate relative position of confidence marker within the remaining string
                    relative_conf_pos = output_lower.find(conf_marker_lower, eval_marker_pos)
                    if relative_conf_pos != -1:
                        eval_end_index = relative_conf_pos - eval_marker_pos - eval_marker_len # Adjust index based on start of substring
                        evaluation_part = evaluation_and_conf[:eval_end_index].strip()
                        confidence_part = evaluation_and_conf[eval_end_index + len(conf_marker_lower):].strip()
                    else: # Should not happen if conf_marker_pos was found, but as fallback:
                        evaluation_part = evaluation_and_conf
                        confidence_part = "Not specified (parse issue)"
                else:
                    # No confidence marker found after evaluation, take whole remaining part as evaluation
                    evaluation_part = evaluation_and_conf
                    confidence_part = "Not specified"
                    
            else: 
                # No Evaluation marker found - less structured output
                # Try to find confidence marker anyway
                if conf_marker_pos != -1:
                        reasoning_part = llm_output_text[:conf_marker_pos].strip()
                        confidence_part = llm_output_text[conf_marker_pos + len(conf_marker_lower):].strip()
                        # Assume the reasoning part might contain the evaluation if no marker found
                        if "correct" in reasoning_part.lower(): evaluation_part = "Potentially Correct (check reasoning)"
                        elif "incorrect" in reasoning_part.lower(): evaluation_part = "Potentially Incorrect (check reasoning)"
                else:
                        # No markers found, treat whole output as reasoning
                        reasoning_part = llm_output_text.strip()
                        # Simple keyword check for evaluation as a last resort
                        if "partially correct" in output_lower: evaluation_part = "Partially Correct (inferred)"
                        elif "correct" in output_lower: evaluation_part = "Correct (inferred)"
                        elif "incorrect" in output_lower: evaluation_part = "Incorrect (inferred)"
                        confidence_part = "Not specified"

        except Exception as parse_ex:
            print(f"Warning: Could not parse Gemini output structure: {parse_ex}")
            # Fallback: keep full output as reasoning, evaluation/confidence as error
            reasoning_part = llm_output_text.strip() 
            evaluation_part = "Parse Error"
            confidence_part = "Parse Error"

        # Assign parsed parts (or defaults/fallbacks)
        # Custom logic to handle inferred evaluations where reasoning might be just the evaluation term
        if "(inferred)" in evaluation_part.lower() and \
            reasoning_part.lower().strip() in ["correct", "incorrect", "partially correct"]:
            llm_reasoning = "LLM did not provide detailed reasoning."
        elif reasoning_part:
            llm_reasoning = reasoning_part
        else:
            llm_reasoning = "No reasoning generated."
        
        llm_evaluation_text = evaluation_part if evaluation_part else "Not specified"
        llm_confidence = confidence_part if confidence_part else "Not specified"
        # -----------------------------------------------------

    except ValueError as ve: # Catch API key error specifically
        print(f"ValueError during Gemini evaluation: {ve}")
        llm_reasoning = "LLM Error: API Key not configured."
        llm_evaluation_text = "LLM Error"
        llm_confidence = "Error"
    except Exception as e:
        print(f"Error during Gemini API evaluation: {e}")
        # Potentially check for specific genai.APIError types if needed
        llm_reasoning = f"Error during LLM reasoning: {type(e).__name__}"
        llm_evaluation_text = "Error during LLM evaluation"
        llm_confidence = "Error"

    # --- Feedback Generation (Refined based on multiple scores) ---
    feedback_parts = []
    confidence = "Medium" # Default confidence

    # Define thresholds (adjust as needed)
    sim_threshold_high = 80
    sim_threshold_med = 60
    bert_threshold_high = 75
    bert_threshold_med = 60
    keyword_threshold_high = 70
    keyword_threshold_med = 40
    concept_threshold_high = 65
    concept_threshold_med = 40
    fluency_threshold_good = 60 # Flesch Reading Ease score
    fluency_threshold_ok = 40

    # --- Core Content Evaluation ---
    if similarity_score >= sim_threshold_high and bert_f1_score >= bert_threshold_high and concept_match_score >= concept_threshold_high:
        feedback_parts.append("Excellent! Your answer demonstrates a strong understanding of the core concepts and matches the expected response well.")
        confidence = "High"
        # Check keywords specifically for excellent answers
        if keyword_score < keyword_threshold_high:
             feedback_parts.append("Consider incorporating more specific keywords to make it even stronger.")

    elif similarity_score >= sim_threshold_med or bert_f1_score >= bert_threshold_med or concept_match_score >= concept_threshold_med:
        feedback_parts.append("Good attempt. Your answer covers some of the main points.")
        confidence = "Medium"
        # Provide more specific feedback for 'Good' attempts
        if concept_match_score < concept_threshold_med and similarity_score >= sim_threshold_med:
             feedback_parts.append("While the overall meaning is similar, focus on including the key underlying concepts more explicitly.")
        elif keyword_score < keyword_threshold_med:
             feedback_parts.append("Try to use more of the relevant keywords associated with this topic.")
        elif similarity_score < sim_threshold_med and bert_f1_score < bert_threshold_med:
             feedback_parts.append("Review the core ideas; the semantic match could be stronger.")
             
    else:
        feedback_parts.append("Your answer seems to diverge significantly from the expected response or misses key aspects.")
        confidence = "Low"
        if keyword_score >= keyword_threshold_med:
            feedback_parts.append("You used some relevant terms, but the overall explanation needs revision.")
        else:
             feedback_parts.append("Focus on understanding the fundamental concepts and associated keywords for this topic.")

    # --- Fluency Feedback ---
    if fluency_score < fluency_threshold_ok:
        feedback_parts.append("Additionally, the clarity and sentence structure could be improved for better readability.")
        # Lower confidence slightly if fluency is very low, regardless of content score
        if confidence == "High": confidence = "Medium"
        elif confidence == "Medium": confidence = "Low"
    elif fluency_score < fluency_threshold_good:
         feedback_parts.append("Consider refining sentence structure for improved fluency.")

    # --- Combine Feedback --- 
    feedback = " ".join(feedback_parts) # Join parts with spaces

    # --- Confidence fallback (if somehow missed) ---
    if confidence is None:
       if similarity_score > 70 or bert_f1_score > 70: confidence = "Medium"
       else: confidence = "Low"
       
    # --- End Refined Feedback Generation ---
    
    # --- Calculate Overall Weighted Score ---
    overall_score = 0.0
    weights = {
        'similarity': 0.20,
        'bert': 0.30,
        'keywords': 0.15,
        'concepts': 0.25,
        'fluency': 0.10
    }
    
    # Ensure scores are valid numbers (default to 0 if not)
    sim_val = similarity_score if isinstance(similarity_score, (int, float)) else 0
    bert_val = bert_f1_score if isinstance(bert_f1_score, (int, float)) else 0
    key_val = keyword_score if isinstance(keyword_score, (int, float)) else 0
    con_val = concept_match_score if isinstance(concept_match_score, (int, float)) else 0
    # Normalize/Cap fluency score? For now, use directly (0-100 range assumed)
    flu_val = fluency_score if isinstance(fluency_score, (int, float)) else 0
    # Clamp fluency score to 0-100 range just in case textstat gives odd values
    flu_val = max(0, min(100, flu_val)) 

    overall_score = (
        sim_val * weights['similarity'] +
        bert_val * weights['bert'] +
        key_val * weights['keywords'] +
        con_val * weights['concepts'] +
        flu_val * weights['fluency']
    )
    overall_score = round(overall_score, 2) # Round to 2 decimal places
    # --- End Overall Score Calculation ---

    # Ensure all scores are Python native types for JSON serialization
    similarity_score_py = float(similarity_score) if similarity_score is not None else 0.0
    bert_f1_score_py = float(bert_f1_score) if bert_f1_score is not None else 0.0
    keyword_score_py = float(keyword_score) if keyword_score is not None else 0.0
    concept_match_score_py = float(concept_match_score) if concept_match_score is not None else 0.0
    fluency_score_py = float(fluency_score) if fluency_score is not None else 0.0
    overall_score_py = float(overall_score) if overall_score is not None else 0.0

    return jsonify({
        "similarity_score": similarity_score_py, 
        "bert_f1_score": bert_f1_score_py,      
        "llm_evaluation": llm_evaluation_text, 
        "llm_reasoning": llm_reasoning,        
        "llm_confidence": llm_confidence,    
        "keyword_score": keyword_score_py, 
        "concept_match_score": concept_match_score_py, 
        "fluency_score": fluency_score_py, 
        "overall_score": overall_score_py, 
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

@app.route('/sessions', methods=['GET'])
def get_saved_sessions():
    """Fetches all saved session documents from Firestore, ordered by creation time."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        sessions_ref = db.collection('savedSessions')
        # Order by creation timestamp, newest first
        query = sessions_ref.order_by('createdAt', direction=firestore.Query.DESCENDING)
        docs = query.stream()

        saved_sessions = []
        for doc in docs:
            session_data = doc.to_dict()
            session_data['id'] = doc.id # Add the document ID
            
            # Convert Firestore Timestamp to ISO 8601 string for JSON compatibility
            if 'createdAt' in session_data and isinstance(session_data['createdAt'], datetime):
                 try:
                     session_data['createdAt'] = session_data['createdAt'].isoformat() + "Z" # Add Z for UTC
                 except Exception as ts_ex:
                     print(f"Warning: Could not format timestamp for session {doc.id}: {ts_ex}")
                     # If formatting fails, convert to string or remove, as Timestamp objects aren't JSON serializable
                     session_data['createdAt'] = str(session_data['createdAt'])
            
            # Optionally process/format other fields if needed before sending
            
            saved_sessions.append(session_data)
        
        print(f"Returning {len(saved_sessions)} saved sessions.")
        return jsonify(saved_sessions)

    except Exception as e:
        print(f"Error fetching saved sessions from Firestore: {e}")
        return jsonify({"error": "Failed to fetch saved sessions from database"}), 500


@app.route('/save_session', methods=['POST'])
def save_session():
    """Saves a new session document to the 'savedSessions' collection in Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500
        
    session_data = request.get_json()

    # Basic validation: Check if essential keys exist (adjust as needed)
    if not session_data or not all(k in session_data for k in ['studentName', 'subject', 'mode', 'results', 'summary']):
        print(f"Warning: Missing required session data fields. Received: {session_data}")
        return jsonify({"error": "Missing required session data fields (studentName, subject, mode, results, summary)"}), 400

    try:
        # Ensure results is a list
        if not isinstance(session_data.get('results'), list):
            return jsonify({"error": "Invalid 'results' format, must be an array."}), 400
        if not isinstance(session_data.get('summary'), dict):
             return jsonify({"error": "Invalid 'summary' format, must be an object."}), 400

        # Add a server timestamp for when the session was saved
        session_data['createdAt'] = firestore.SERVER_TIMESTAMP 

        # Reference the collection
        sessions_ref = db.collection('savedSessions')
        
        # Add the session data as a new document with an auto-generated ID
        update_time, doc_ref = sessions_ref.add(session_data)
        
        session_id = doc_ref.id
        print(f"Saved session with ID: {session_id}")
        
        # Return success message and the ID of the saved session
        return jsonify({"message": "Session saved successfully to Firestore", "sessionId": session_id}), 201

    except Exception as e:
        print(f"Error saving session to Firestore: {e}")
        return jsonify({"error": "Failed to save session to database"}), 500

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

# === Admin Routes === 

# --- Admin: Subject Management ---

@app.route('/admin/subjects', methods=['POST'])
@check_auth # Apply decorator
def admin_add_subject():
    """Adds a new subject document to Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        data = request.get_json()
        subject_name = data.get('subject')

        if not subject_name or not isinstance(subject_name, str) or not subject_name.strip():
            return jsonify({"error": "Invalid or missing 'subject' name in request body"}), 400
        
        subject_name = subject_name.strip()

        # Check if subject already exists (case-sensitive query)
        subjects_ref = db.collection('vivaQuestions')
        query = subjects_ref.where('subject', '==', subject_name).limit(1)
        existing = list(query.stream()) # Execute query

        if len(existing) > 0:
             return jsonify({"error": f"Subject '{subject_name}' already exists."}), 409 # Conflict

        # Data for the new subject document
        new_subject_data = {
            'subject': subject_name,
            'questions': [], # Initialize with an empty questions array
            'createdAt': firestore.SERVER_TIMESTAMP # Optional timestamp
        }

        # Add a new document with an auto-generated ID
        update_time, doc_ref = subjects_ref.add(new_subject_data)
        
        # Prepare data for the JSON response, excluding or converting server timestamp
        response_data = {
            'id': doc_ref.id,
            'subject': subject_name,
            'questions': [] # Match the structure expected by client if it uses questions array
        }
        # If you needed to return the timestamp, you would convert update_time (which is a datetime object after write)
        # response_data['createdAt'] = update_time.isoformat() 

        print(f"Added new subject '{subject_name}' with ID: {doc_ref.id}")
        return jsonify(response_data), 201 # 201 Created status

    except Exception as e:
        print(f"Error adding subject to Firestore: {e}")
        return jsonify({"error": "Failed to add subject to database"}), 500

# --- Admin: Question Management (within subjects) ---

@app.route('/admin/questions', methods=['GET'])
@check_auth # Apply decorator
def admin_get_all_questions():
    """Fetches all subjects and their questions from Firestore for the admin panel."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        questions_ref = db.collection('vivaQuestions').stream()
        all_subjects_data = [] 
        for doc in questions_ref:
            data = doc.to_dict()
            # Ensure essential fields exist
            subject_name = data.get('subject')
            questions_list = data.get('questions', []) 
            
            if subject_name: # Only include if subject name is present
                # Ensure all questions in the list have IDs (add if missing - legacy?) - IMPORTANT for EDIT/DELETE
                processed_questions = []
                needs_update = False
                for q in questions_list:
                    if isinstance(q, dict) and 'id' not in q:
                        q['id'] = str(uuid.uuid4()) # Assign a new ID
                        needs_update = True 
                        print(f"Warning: Assigned new ID {q['id']} to question in subject {subject_name}")
                    if isinstance(q, dict): # Only add valid question dicts
                        processed_questions.append(q)
                    else:
                        print(f"Warning: Skipped invalid item in questions array for subject {subject_name}: {q}")
                
                # If any question IDs were added, update the document in Firestore
                if needs_update:
                    try:
                        db.collection('vivaQuestions').document(doc.id).update({'questions': processed_questions})
                        print(f"Updated Firestore document {doc.id} with new question IDs.")
                    except Exception as update_err:
                        print(f"Error updating document {doc.id} with new question IDs: {update_err}")
                        # Continue processing other subjects even if one update fails

                all_subjects_data.append({
                    'id': doc.id, # Document ID (Subject ID)
                    'subject': subject_name,
                    'questions': processed_questions # Use the processed list with IDs
                })
            else:
                print(f"Warning: Document {doc.id} in vivaQuestions missing 'subject' field.")

        print(f"Returning admin data for {len(all_subjects_data)} subjects.")
        return jsonify(all_subjects_data) # Return the list of subject objects

    except Exception as e:
        print(f"Error fetching admin questions data from Firestore: {e}")
        return jsonify({"error": "Failed to fetch admin questions data from database"}), 500

@app.route('/admin/questions/<subject_id>', methods=['POST'])
@check_auth # Apply decorator
def admin_add_question(subject_id):
    """Adds a new question to the 'questions' array of a specific subject document."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500
    if not subject_id:
        return jsonify({"error": "Missing subject_id parameter"}), 400

    try:
        data = request.get_json()
        # Validate incoming question data
        question_text = data.get('question')
        expected_text = data.get('expected')
        keywords_list = data.get('keywords', []) # Default to empty list

        if not all([question_text, expected_text]) or not isinstance(question_text, str) or not isinstance(expected_text, str):
            return jsonify({"error": "Invalid or missing 'question' or 'expected' text"}), 400
        if not isinstance(keywords_list, list):
             return jsonify({"error": "Invalid 'keywords' format, must be an array"}), 400
        
        # Ensure keywords are strings and stripped
        valid_keywords = [str(k).strip() for k in keywords_list if isinstance(k, (str, int, float)) and str(k).strip()]

        # Create the new question object with a unique ID
        new_question_object = {
            'id': str(uuid.uuid4()), # Generate and add unique ID
            'question': question_text.strip(),
            'expected': expected_text.strip(),
            'keywords': valid_keywords
        }

        doc_ref = db.collection('vivaQuestions').document(subject_id)

        # Atomically add the new question object to the 'questions' array
        # This ensures the operation is safe even with concurrent requests
        doc_ref.update({
            'questions': firestore.ArrayUnion([new_question_object])
        })
        
        print(f"Added question to subject ID: {subject_id} (Question ID: {new_question_object['id']})")
        # Return success message and the new question object (including its new ID)
        return jsonify(new_question_object), 201 # 201 Created status

    except Exception as e:
        # Check if the error is because the document doesn't exist (less likely with ArrayUnion)
        # But could be other issues like permissions or malformed data.
        print(f"Error adding question to subject {subject_id} in Firestore: {e}")
        return jsonify({"error": "Failed to add question to database"}), 500

@app.route('/admin/questions/<subject_id>/<question_id>', methods=['PUT'])
@check_auth # Apply decorator
def admin_update_question(subject_id, question_id):
    """Updates a specific question within the 'questions' array of a subject document."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500
    if not subject_id or not question_id:
        return jsonify({"error": "Missing subject_id or question_id parameter"}), 400

    try:
        # Get updated data from request body
        data = request.get_json()
        question_text = data.get('question')
        expected_text = data.get('expected')
        keywords_list = data.get('keywords') # Allow None or empty list

        # Validate incoming data
        if not all([question_text, expected_text]) or not isinstance(question_text, str) or not isinstance(expected_text, str):
            return jsonify({"error": "Invalid or missing 'question' or 'expected' text"}), 400
        # Allow keywords to be None or empty list, but if present, must be a list
        if keywords_list is not None and not isinstance(keywords_list, list):
             return jsonify({"error": "Invalid 'keywords' format, must be an array or null"}), 400
        
        # Process keywords if provided, otherwise default to empty list
        valid_keywords = []
        if isinstance(keywords_list, list):
            valid_keywords = [str(k).strip() for k in keywords_list if isinstance(k, (str, int, float)) and str(k).strip()]

        # Transaction recommended for read-modify-write on array elements
        @firestore.transactional
        def update_question_in_transaction(transaction, doc_ref):
            doc_snapshot = doc_ref.get(transaction=transaction)
            if not doc_snapshot.exists:
                raise FileNotFoundError(f"Subject document with ID {subject_id} not found")

            subject_data = doc_snapshot.to_dict()
            questions_array = subject_data.get('questions', [])
            updated_questions_array = []
            question_found = False
            updated_question_data = None

            # Iterate and rebuild the array
            for question in questions_array:
                if isinstance(question, dict) and question.get('id') == question_id:
                    # Found the question, update its content but keep the ID
                    updated_question = {
                        'id': question_id, # Keep original ID
                        'question': question_text.strip(),
                        'expected': expected_text.strip(),
                        'keywords': valid_keywords
                    }
                    updated_questions_array.append(updated_question)
                    question_found = True
                    updated_question_data = updated_question # Store data to return
                elif isinstance(question, dict): # Only keep valid dicts
                    # Keep other questions as they are
                    updated_questions_array.append(question)

            if not question_found:
                raise ValueError(f"Question with ID {question_id} not found in subject {subject_id}")

            # Update the document within the transaction
            transaction.update(doc_ref, {
                'questions': updated_questions_array
            })
            return updated_question_data # Return the updated data from transaction

        # --- Execute Transaction --- 
        doc_reference = db.collection('vivaQuestions').document(subject_id)
        result_data = update_question_in_transaction(db.transaction(), doc_reference)
        # --- End Transaction --- 
        
        print(f"Updated question ID {question_id} in subject ID: {subject_id}")
        # Return the updated question data retrieved from the transaction
        return jsonify(result_data), 200

    except FileNotFoundError as e: # Catch specific error from transaction
         print(f"Error updating question: {e}")
         return jsonify({"error": str(e)}), 404
    except ValueError as e: # Catch specific error from transaction
         print(f"Error updating question: {e}")
         return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"Error updating question {question_id} in subject {subject_id}: {e}")
        return jsonify({"error": "Failed to update question in database"}), 500

@app.route('/admin/questions/<subject_id>/<question_id>', methods=['DELETE'])
@check_auth # Apply decorator
def admin_delete_question(subject_id, question_id):
    """Removes a specific question object from the 'questions' array of a subject document."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500
    if not subject_id or not question_id:
        return jsonify({"error": "Missing subject_id or question_id parameter"}), 400

    # Transaction recommended for read-modify-delete on array elements
    @firestore.transactional
    def delete_question_in_transaction(transaction, doc_ref):
        doc_snapshot = doc_ref.get(transaction=transaction)
        if not doc_snapshot.exists:
            raise FileNotFoundError(f"Subject document with ID {subject_id} not found")

        subject_data = doc_snapshot.to_dict()
        questions_array = subject_data.get('questions', [])

        # Find the question object to remove based on its ID
        question_to_remove = None
        for question in questions_array:
            if isinstance(question, dict) and question.get('id') == question_id:
                question_to_remove = question
                break
        
        if question_to_remove is None:
             # Question with the given ID wasn't found in the array
             raise ValueError(f"Question with ID {question_id} not found in subject {subject_id}")

        # Atomically remove the specific question object from the array using ArrayRemove
        transaction.update(doc_ref, {
            'questions': firestore.ArrayRemove([question_to_remove])
        })
        # No data needs to be returned from the transaction itself for delete

    try:
        # --- Execute Transaction --- 
        doc_reference = db.collection('vivaQuestions').document(subject_id)
        delete_question_in_transaction(db.transaction(), doc_reference)
        # --- End Transaction --- 
        
        print(f"Deleted question ID {question_id} from subject ID: {subject_id}")
        return jsonify({"message": "Question deleted successfully"}), 200 # 200 OK status

    except FileNotFoundError as e: # Catch specific error from transaction
         print(f"Error deleting question: {e}")
         return jsonify({"error": str(e)}), 404
    except ValueError as e: # Catch specific error from transaction
         print(f"Error deleting question: {e}")
         return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"Error deleting question {question_id} from subject {subject_id}: {e}")
        return jsonify({"error": "Failed to delete question from database"}), 500

# --- Admin: Scenario Management ---

@app.route('/admin/scenarios', methods=['GET'])
@check_auth # Apply decorator
def admin_get_all_scenarios():
    """Fetches all scenarios from the 'scenarios' collection in Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        scenarios_ref = db.collection('scenarios').stream() # Assuming collection name is 'scenarios'
        all_scenarios = []
        for doc in scenarios_ref:
            scenario_data = doc.to_dict()
            scenario_data['id'] = doc.id # Add the document ID
            all_scenarios.append(scenario_data)
        
        print(f"Returning {len(all_scenarios)} scenarios from Firestore.")
        return jsonify(all_scenarios)
    except Exception as e:
        print(f"Error fetching scenarios from Firestore: {e}")
        return jsonify({"error": "Failed to fetch scenarios from database"}), 500

@app.route('/admin/scenarios', methods=['POST'])
@check_auth # Apply decorator
def admin_add_scenario():
    """Adds a new scenario document to the 'scenarios' collection in Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        data = request.get_json()
        scenario_text = data.get('scenario')

        if not scenario_text or not isinstance(scenario_text, str) or not scenario_text.strip():
            return jsonify({"error": "Invalid or missing 'scenario' text in request body"}), 400

        # Data to be added to Firestore
        new_scenario_data = {
            'scenario': scenario_text.strip(), # Store the trimmed text
            'createdAt': firestore.SERVER_TIMESTAMP # Optional: add creation timestamp
        }

        # Add a new document with an auto-generated ID
        update_time, doc_ref = db.collection('scenarios').add(new_scenario_data)
        
        new_scenario_data['id'] = doc_ref.id # Add ID to response
        print(f"Added new scenario with ID: {doc_ref.id}")
        # Return success message and the data of the new document (including ID)
        return jsonify(new_scenario_data), 201 # 201 Created status

    except Exception as e:
        print(f"Error adding scenario to Firestore: {e}")
        return jsonify({"error": "Failed to add scenario to database"}), 500

@app.route('/admin/scenarios/<scenario_id>', methods=['PUT'])
@check_auth # Apply decorator
def admin_update_scenario(scenario_id):
    """Updates an existing scenario document in Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500
    if not scenario_id:
        return jsonify({"error": "Missing scenario_id parameter"}), 400

    try:
        data = request.get_json()
        updated_scenario_text = data.get('scenario')

        if not updated_scenario_text or not isinstance(updated_scenario_text, str) or not updated_scenario_text.strip():
            return jsonify({"error": "Invalid or missing 'scenario' text in request body"}), 400

        doc_ref = db.collection('scenarios').document(scenario_id)
        
        # Check if document exists before updating (more robust)
        doc_snapshot = doc_ref.get()
        if not doc_snapshot.exists:
            return jsonify({"error": f"Scenario with ID {scenario_id} not found"}), 404
            
        # Use update() to modify specific fields
        doc_ref.update({
            'scenario': updated_scenario_text.strip(),
            'updatedAt': firestore.SERVER_TIMESTAMP # Optional: add update timestamp
        })
        
        print(f"Updated scenario with ID: {scenario_id}")
        # Return success message and the updated data
        updated_data = {'id': scenario_id, 'scenario': updated_scenario_text.strip()}
        return jsonify(updated_data), 200 # 200 OK status

    except Exception as e:
        print(f"Error updating scenario {scenario_id} in Firestore: {e}")
        return jsonify({"error": "Failed to update scenario in database"}), 500

@app.route('/admin/scenarios/<scenario_id>', methods=['DELETE'])
@check_auth # Apply decorator
def admin_delete_scenario(scenario_id):
    """Deletes a scenario document from Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500
    if not scenario_id:
        return jsonify({"error": "Missing scenario_id parameter"}), 400

    try:
        doc_ref = db.collection('scenarios').document(scenario_id)
        
        # Check if it exists before attempting delete (optional, delete is idempotent)
        # doc_snapshot = doc_ref.get()
        # if not doc_snapshot.exists:
        #     return jsonify({"error": f"Scenario with ID {scenario_id} not found"}), 404
            
        doc_ref.delete() # Deletes the document. Does not raise error if not found.
        
        print(f"Attempted delete for scenario with ID: {scenario_id}")
        # Return success message
        return jsonify({"message": "Scenario deleted successfully"}), 200 # 200 OK status

    except Exception as e:
        print(f"Error deleting scenario {scenario_id} from Firestore: {e}")
        return jsonify({"error": "Failed to delete scenario from database"}), 500

# === User Signup Route ===

@app.route('/signup', methods=['POST'])
def signup_user():
    """Registers a new user with email/password and stores info in Firestore."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name', '') # Optional name

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    try:
        # Create user in Firebase Authentication
        user_record = auth.create_user(
            email=email,
            password=password,
            display_name=name # Set display name if provided
        )
        print(f"Successfully created new user: {user_record.uid}")

        # Store user info in Firestore database
        user_doc_ref = db.collection('users').document(user_record.uid)
        user_doc_ref.set({
            'email': email,
            'name': name,
            'role': 'student', # Default role for new signups
            'created_at': firestore.SERVER_TIMESTAMP
        })
        print(f"Stored user info in Firestore for user: {user_record.uid}")

        # Return success response (maybe just the UID or a success message)
        return jsonify({"message": "User created successfully", "userId": user_record.uid}), 201

    except auth.EmailAlreadyExistsError:
        print(f"Signup failed: Email {email} already exists.")
        return jsonify({"error": "Email already exists"}), 409 # Conflict
    except auth.FirebaseAuthError as e:
        # Catch other Firebase auth errors (e.g., weak password)
        print(f"Firebase Auth error during signup: {e}")
        # You might want to parse the specific error for a better message
        return jsonify({"error": f"Authentication error: {e}"}), 400
    except Exception as e:
        print(f"Error during signup process: {e}")
        # Attempt to delete the auth user if Firestore write failed (optional cleanup)
        try:
            if 'user_record' in locals() and user_record:
                auth.delete_user(user_record.uid)
                print(f"Cleaned up Firebase Auth user {user_record.uid} due to Firestore error.")
        except Exception as cleanup_ex:
             print(f"Error during signup cleanup: {cleanup_ex}")
        
        return jsonify({"error": "An internal error occurred during signup."}), 500

# === Scenario Route for Students ===
@app.route('/scenarios', methods=['GET'])
def get_all_scenarios_for_students():
    """Fetches all scenarios from the 'scenarios' collection in Firestore for student practice."""
    if not db:
        return jsonify({"error": "Database not initialized"}), 500

    try:
        scenarios_ref = db.collection('scenarios').stream() # Assuming collection name is 'scenarios'
        all_scenarios = []
        for doc in scenarios_ref:
            scenario_data = doc.to_dict()
            scenario_data['id'] = doc.id # Add the document ID
            all_scenarios.append(scenario_data)
        
        print(f"Returning {len(all_scenarios)} scenarios for student practice.")
        return jsonify(all_scenarios)
    except Exception as e:
        print(f"Error fetching scenarios for students from Firestore: {e}")
        return jsonify({"error": "Failed to fetch scenarios from database"}), 500

if __name__ == '__main__':
    app.run(debug=True)
