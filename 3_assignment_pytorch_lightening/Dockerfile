# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 6006 available for TensorBoard
EXPOSE 6006

# Set the default command to python
CMD ["python"]
