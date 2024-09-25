# Use a slim version of Python as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app


# Copy all Python files
COPY model.py train.py eval.py infer.py  /app/

# # Copy project files
# COPY model/ /app/model/
# COPY train/ /app/train/
# COPY eval/ /app/eval/
# COPY infer/ /app/infer/

# Copy requirements file and install dependencies
COPY requirements.txt .

#Install system dependencies for PyTorch and other libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy kaggle.json for authentication (ensure you have this file)
COPY kaggle.json /root/.kaggle/kaggle.json

# Ensure permissions for kaggle.json
RUN chmod 600 /root/.kaggle/kaggle.json

# # Create directories for data and model checkpoints
# RUN mkdir -p /data /app/model_checkpoints

# Create a volume for data and model storage
VOLUME ["/data", "/app/model"]

# Entrypoint left empty to allow flexibility for each script
ENTRYPOINT [""]
