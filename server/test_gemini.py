import os
print(f"Attempting to import google.generativeai. OS version: {os.name}")
try:
    import google.generativeai as genai
    print(f"Successfully imported google.generativeai.")
    print(f"SDK Version: {genai.__version__}")
    
    # Attempt to access the GenerativeModel class
    model = genai.GenerativeModel(model_name='gemini-pro') # Using a common model name for this test
    print("Successfully accessed genai.GenerativeModel and instantiated a model.")
    
except ImportError as ie:
    print(f"ImportError: {ie}. The library might not be installed correctly.")
except AttributeError as ae:
    print(f"AttributeError: {ae}. This is the error your Flask app is getting.")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 