# Development
./deploy-dev.sh

# Production  
./deploy-prod.sh

# Manual with custom env
cp .env.custom .env
docker-compose up -d --build

# Ghavio1516 - Z2K