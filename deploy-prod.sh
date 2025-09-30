#!/bin/bash
# deploy-prod.sh - Production deployment script

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

# Check if .env.prod exists
if [ ! -f ".env.prod" ]; then
    print_error ".env.prod file not found!"
    echo ""
    echo "Creating .env.prod from template..."
    
    # Get current directory
    CURRENT_DIR=$(pwd)
    
    # Create .env.prod
    cat > .env.prod << EOF
# .env.prod - Production Environment (Auto-generated)
PROJECT_NAME=deptech-ocr-prod
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
AUTO_RELOAD=false

# Health Check - Production settings
HEALTH_CHECK_INTERVAL=30s
HEALTH_CHECK_TIMEOUT=10s
HEALTH_CHECK_RETRIES=3
EOF
    
    print_success ".env.prod created with current path: ${CURRENT_DIR}"
fi

# Copy .env.prod to .env
print_status "Setting up production environment..."
cp .env.prod .env
print_success "Environment configured for production"

# Create required directories
print_status "Creating required directories..."
mkdir -p app uploads logs
print_success "Directories created"

# Stop any existing containers
print_status "Stopping existing containers..."
docker compose down 2>/dev/null || true
print_success "Existing containers stopped"

# Build and start production environment
print_status "Building and starting production containers..."
print_warning "This may take several minutes..."

docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

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
echo "🔧 Configuration:"
echo "   • Language: ch (optimized for Indonesian)"
echo "   • Quality: high (maximum accuracy)"
echo "   • GPU: enabled (if available)"
echo ""

# Test API endpoint
print_status "Testing API endpoint..."
sleep 10

if curl -s http://localhost:2005/health > /dev/null; then
    print_success "API is responding!"
    echo ""
    echo "✅ Production deployment complete!"
else
    print_warning "API not responding yet - check logs"
    echo "   Use: docker compose logs -f"
fi
