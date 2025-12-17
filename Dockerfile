# Multi-stage Dockerfile for FPL Manager Agent

# Stage 1: Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose the port (Note: This is mostly for documentation in cloud builds)
EXPOSE 8501

# Health check updated to look at the dynamic port (or default 8501)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/health || exit 1

# Run FastAPI application using the shell form to allow variable expansion
# We use ${PORT:-8501} to use the cloud's port, or default to 8501 if locally run
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8501}"