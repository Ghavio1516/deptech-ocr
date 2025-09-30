#!/bin/bash
# deploy-prod.sh - Production deployment script (Fixed)

set -e  # Exit on any error

echo "🚀 Deploying Deptech OCR - Production Environment"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get current directory
CURRENT_DIR=$(pwd)
print_status "Working directory: ${CURRENT_DIR}"

# Create .env.prod if it doesn't exist
if [ ! -f ".env.prod" ]; then
    print_warning ".env.prod not found, creating it..."
    
    cat > .env.prod << 'EOF'
# .env.prod - Production Environment
PROJECT_NAME=deptech-ocr-prod

# API Configuration
API_PORT=2005
API_HOST=0.0.0.0

# OCR Configuration - Production optimized
DEFAULT_LANGUAGE=ch
DEFAULT_QUALITY=high
DEFAULT_EXTRACTION_MODE=hybrid
ENABLE_CLEANSING=true

# Performance - Maximum for production
OCR_CONFIDENCE_THRESHOLD=0.6
MAX_IMAGE_SIZE_MB=100
ENABLE_GPU=true

# Environment
ENV=production
TZ=Asia/Jakarta
DEBUG=false

# Health Check - Production settings
HEALTH_CHECK_INTERVAL=30s
HEALTH_CHECK_TIMEOUT=10s
HEALTH_CHECK_RETRIES=3
EOF
    
    print_success ".env.prod created"
fi

# Copy .env.prod to .env
print_status "Setting up production environment..."
cp .env.prod .env
print_success "Environment configured for production"

# Create required directories
print_status "Creating required directories..."
mkdir -p app uploads logs
touch uploads/.gitkeep logs/.gitkeep

# Ensure requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_warning "requirements.txt not found, creating it..."
    cat > requirements.txt << 'EOF'
fastapi[standard]>=0.113.0,<0.114.0
uvicorn>=0.30.0
pydantic>=2.0.0
pillow>=10.0.0
paddlepaddle>=3.2.0
paddleocr>=3.2.0
opencv-python-headless>=4.10.0
numpy>=1.21.0
PyMuPDF>=1.23.0
python-docx>=1.1.0
python-dotenv>=1.0.0
EOF
fi

# Ensure app/main.py exists
if [ ! -f "app/main.py" ]; then
    print_error "app/main.py not found!"
    print_status "Please ensure your main.py file is in the app/ directory"
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    print_warning "Dockerfile not found, creating basic production Dockerfile..."
    cat > Dockerfile << 'EOF'
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

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["fastapi", "run", "app/main.py", "--port", "8000", "--host", "0.0.0.0"]
EOF
    
    print_success "Dockerfile created"
fi

print_success "Directory structure verified"

# Stop any existing containers
print_status "Stopping existing containers..."
docker compose down 2>/dev/null || true
print_success "Existing containers stopped"

# Build and start production environment
print_status "Building and starting production containers..."
print_warning "This may take several minutes on first build..."

# Use the fixed compose files
docker compose -f docker-compose.prod.yml up -d --build

# Wait for container to be ready
print_status "Waiting for container to start..."
sleep 15

# Check if container is running
if docker compose ps | grep -q "deptech-ocr-prod"; then
    print_success "Container started successfully!"
else
    print_error "Container failed to start"
    echo ""
    echo "Checking logs:"
    docker compose logs --tail=20
    exit 1
fi

echo ""
echo "🎉 Production environment deployed successfully!"
echo "=============================================="
echo ""
echo "📊 Service Information:"
echo "   • Container: deptech-ocr-prod"
echo "   • Environment: production"
echo "   • Port: 2005"
echo ""
echo "🌐 Access URLs:"
echo "   • API Endpoint: http://localhost:2005"
echo "   • Health Check: http://localhost:2005/health"
echo ""
echo "🔧 Production Configuration:"
echo "   • Language: ch (optimized for Indonesian)"
echo "   • Quality: high (maximum accuracy)"
echo "   • GPU: enabled (if hardware supports)"
echo "   • Confidence: 0.6 (high accuracy threshold)"
echo ""
echo "📋 Management Commands:"
echo "   • View logs: docker compose logs -f"
echo "   • Restart: docker compose restart"
echo "   • Stop: docker compose down"
echo "   • Stats: docker compose stats"
echo ""

# Test API endpoint
print_status "Testing API endpoint in 30 seconds..."
sleep 30

if curl -s http://localhost:2005/health > /dev/null 2>&1; then
    print_success "API is responding!"
    echo ""
    echo "✅ Production deployment complete and ready!"
    echo ""
    print_status "Monitoring: docker compose logs -f"
else
    print_warning "API not responding yet - may still be initializing"
    echo ""
    echo "Check container status: docker compose ps"
    echo "View logs: docker compose logs -f"
    echo "Wait a few more minutes for initialization..."
fi

echo ""
print_success "Production environment is running! 🚀"
