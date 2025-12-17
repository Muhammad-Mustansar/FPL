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

# Expose no fixed port; container will bind to the PORT env provided by Render
EXPOSE 8000 8501

# Health check: attempt the configured PORT (fallback to 8000 then 8501)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
        CMD sh -c 'for p in "${PORT:-}" 8000 8501; do if [ -n "$p" ]; then curl -fsS http://localhost:$p/health && exit 0 || true; fi; done; exit 1'

# Allow running either the backend (FastAPI) or frontend (Streamlit) from the same image
# Use SERVICE_ROLE env var: 'backend' (default) or 'frontend'
CMD sh -c '\
    if [ "${SERVICE_ROLE:-backend}" = "frontend" ]; then \
        echo "Starting Streamlit frontend on port ${PORT:-8501}"; \
        streamlit run frontend/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.enableXsrfProtection false; \
    else \
        echo "Starting FastAPI backend on port ${PORT:-8000}"; \
        uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}; \
    fi'