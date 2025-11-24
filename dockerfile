# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary files/folders
COPY app/ ./app/
COPY src/ ./src/
COPY models/model.pkl ./models/model.pkl
COPY models/preprocessor.pkl ./models/preprocessor.pkl
COPY app/reports ./app/reports

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
