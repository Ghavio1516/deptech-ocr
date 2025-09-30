# Make scripts executable
chmod +x deploy-dev.sh deploy-prod.sh

# Development (with hot reload)
./deploy-dev.sh

# Production (optimized build)
./deploy-prod.sh

# View logs
docker compose logs -f

# Stop services
docker compose down

# Check status
docker compose ps

Feature        |  Development      |  Production     
---------------+-------------------+-----------------
Quality        |  balanced         |  high           
GPU            |  disabled         |  enabled        
Confidence     |  0.4              |  0.6            
Max File Size  |  20MB             |  100MB          
Hot Reload     |  ✅                |  ❌              
Debug Mode     |  ✅                |  ❌              
Build Method   |  Runtime install  |  Pre-built image