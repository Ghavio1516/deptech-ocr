#!/bin/bash
# deploy-dev.sh - Development deployment script

set -e  # Exit on any error

echo "🚀 Deploying Deptech OCR - Development Environment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if .env.dev exists
if [ ! -f ".env.dev" ]; then
    print_error ".env.dev file not found!"
    echo ""
    echo "Creating .env.dev from template..."
    
    # Get current directory
    CURRENT_DIR=$(pwd)
    
    # Create .env.dev
    cat > .env.dev << EOF
# .env.dev - Development Environment (Auto-generated)
PROJECT_NAME=deptech-ocr-dev
PROJECT_BASE_PATH=${CURRENT_DIR}

# Docker Volume Mappings
APP_PATH=\${PROJECT_BASE_PATH}/app
UPLOADS_PATH=\${PROJECT_BASE_PATH}/uploads
LOGS_PATH=\${PROJECT_BASE_PATH}/logs
REQUIREMENTS_PATH=\${PROJECT_BASE_PATH}/requirements.txt

# Container Mappings
CONTAINER_APP_PATH=/code/app
CONTAINER_UPLOADS_PATH=/uploads
CONTAINER_LOGS_PATH=/logs
CONTAINER_REQUIREMENTS_PATH=/code/requirements.txt

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
AUTO_RELOAD=true

# Health Check - Faster for dev
HEALTH_CHECK_INTERVAL=15s
HEALTH_CHECK_TIMEOUT=5s
HEALTH_CHECK_RETRIES=2
EOF
    
    print_success ".env.dev created with current path: ${CURRENT_DIR}"
fi

# Copy .env.dev to .env
print_status "Setting up development environment..."
cp .env.dev .env
print_success "Environment configured for development"

# Create required directories
print_status "Creating required directories..."
mkdir -p app uploads logs
touch uploads/.gitkeep logs/.gitkeep
print_success "Directories created"

# Stop any existing containers
print_status "Stopping existing containers..."
docker compose down 2>/dev/null || true
print_success "Existing containers stopped"

# Build and start development environment
print_status "Building and starting development containers..."
print_warning "This may take a few minutes on first run..."

# Use the new docker compose command (not docker-compose)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

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
echo "🔧 Configuration:"
echo "   • Language: ch (Chinese model for Indonesian text)"
echo "   • Quality: balanced (faster for development)"
echo "   • GPU: disabled"
echo ""

# Test API endpoint
print_status "Testing API endpoint..."
sleep 5

if curl -s http://localhost:2005/health > /dev/null; then
    print_success "API is responding!"
    echo ""
    echo "✅ Deployment complete and ready for development!"
else
    print_warning "API not responding yet - may still be initializing"
    echo "   Check logs with: docker compose logs -f"
fi

echo ""
print_status "To view real-time logs: docker compose logs -f"
