# Use a slim Python image for efficiency
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies needed for some ML libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including the app folder) into the container
COPY . .

# Set environment variables to ensure internal imports work
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Render provides a dynamic port via the $PORT environment variable.
# We expose a default but the command below will use Render's port.
EXPOSE 10000

# FIX: Change 'main:app' to 'app.main:app' because main.py is inside the app folder.
# We use sh -c to ensure the $PORT variable provided by Render is correctly read.
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"