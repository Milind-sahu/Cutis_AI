# Dockerfile for Hugging Face Spaces
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/trained_model/SKIN_MODEL_BEST.keras \
    TF_CPP_MIN_LOG_LEVEL=2

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy backend app code
COPY backend/app ./app

# Copy the trained model
# Note: Ensure the model is tracked by Git LFS before pushing
COPY backend/trained_model ./trained_model

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
