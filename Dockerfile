# Dockerfile for Production
FROM python:3.10-slim

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /code/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app/ /code/app/

# Create uploads and logs directories
RUN mkdir -p /uploads /logs

# Set environment variables
ENV PYTHONPATH=/code
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["fastapi", "run", "app/main.py", "--port", "8000", "--host", "0.0.0.0"]
