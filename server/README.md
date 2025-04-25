# Server Setup and Configuration

This server application uses Flask and integrates with Firebase Admin SDK and various NLP models. To keep sensitive information secure, we load credentials and API keys from environment variables.

## Prerequisites
- Python 3.9 or newer
- `pip` package manager

## Environment Variables
Create a `.env` file in the `server` directory (this file is ignored by Git) with the following variables:

```bash
# Path to the Firebase service account key JSON file
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service_account_key.json
```

## Installation

1. Navigate to the server directory:
   ```bash
   cd server
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server
```bash
flask run --host=0.0.0.0 --port=5000
```

## Security Best Practices
- Never commit your `.env` file or service account key to the repository.
- Use a secrets manager for production environments to manage sensitive credentials.
- Ensure the `.gitignore` file includes `.env` and key files. 