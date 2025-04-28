# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install any system dependencies (optional, adjust if needed)
RUN apt-get update && apt-get install -y build-essential --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 5000

# Start the app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT"] 