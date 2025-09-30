#!/bin/bash
# deploy-dev.sh - Development deployment script (Fixed)

set -e  # Exit on any error

echo "🚀 Deploying Deptech OCR - Development Environment"
echo "=================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
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

# Create .env.dev if it doesn't exist
if [ ! -f ".env.dev" ]; then
    print_warning ".env.dev not found, creating it..."
    
    cat > .env.dev << 'EOF'
# .env.dev - Development Environment
PROJECT_NAME=deptech-ocr-dev

# API Configuration
API_PORT=2005
API_HOST=0.0.0.0

# OCR Configuration - Development optimized
DEFAULT_LANGUAGE=ch
DEFAULT_QUALITY=balanced
DEFAULT_EXTRACTION_MODE=hybrid
ENABLE_CLEANSING=true

# Performance - Lower for development
OCR_CONFIDENCE_THRESHOLD=0.4
MAX_IMAGE_SIZE_MB=20
ENABLE_GPU=false

# Environment
ENV=development
TZ=Asia/Jakarta
DEBUG=true

# Health Check - Faster for dev
HEALTH_CHECK_INTERVAL=15s
HEALTH_CHECK_TIMEOUT=5s
HEALTH_CHECK_RETRIES=2
EOF
    
    print_success ".env.dev created"
fi

# Copy .env.dev to .env
print_status "Setting up development environment..."
cp .env.dev .env
print_success "Environment configured for development"

# Create required directories
print_status "Creating required directories..."
mkdir -p app uploads logs
touch uploads/.gitkeep logs/.gitkeep

# Ensure requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_warning "requirements.txt not found, creating basic one..."
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

# Ensure app/main.py exists (basic check)
if [ ! -f "app/main.py" ]; then
    print_error "app/main.py not found!"
    print_status "Please ensure your main.py file is in the app/ directory"
    exit 1
fi

print_success "Directory structure verified"

# Stop any existing containers
print_status "Stopping existing containers..."
docker compose down 2>/dev/null || true
print_success "Existing containers stopped"

# Build and start development environment
print_status "Building and starting development containers..."
print_warning "This may take a few minutes on first run..."

# Use the fixed compose files with relative paths
docker compose -f docker-compose.dev.yml up -d --build

# Wait for container to be ready
print_status "Waiting for container to start..."
sleep 10

# Check if container is running
if docker compose ps | grep -q "deptech-ocr-dev"; then
    print_success "Container started successfully!"
else
    print_error "Container failed to start"
    echo ""
    echo "Checking logs:"
    docker compose logs --tail=20
    exit 1
fi

# Display status
echo ""
echo "🎉 Development environment deployed successfully!"
echo "=============================================="
echo ""
echo "📊 Service Information:"
echo "   • Container: deptech-ocr-dev"
echo "   • Environment: development"
echo "   • Port: 2005"
echo ""
echo "🌐 Access URLs:"
echo "   • API Endpoint: http://localhost:2005"
echo "   • Health Check: http://localhost:2005/health"
echo "   • API Docs: http://localhost:2005/docs"
echo ""
echo "📋 Useful Commands:"
echo "   • View logs: docker compose logs -f"
echo "   • Restart: docker compose restart"
echo "   • Stop: docker compose down"
echo "   • Shell access: docker compose exec deptech-ocr bash"
echo ""

# Test API endpoint
print_status "Testing API endpoint in 30 seconds..."
sleep 30

if curl -s http://localhost:2005/health > /dev/null 2>&1; then
    print_success "API is responding!"
    echo ""
    echo "✅ Deployment complete and ready for development!"
else
    print_warning "API not responding yet - may still be initializing"
    echo ""
    echo "Check container status: docker compose ps"
    echo "View logs: docker compose logs -f"
fi

echo ""
print_status "Happy coding! 🚀"
