#!/bin/bash
echo "🚀 Starting Production Environment..."
cp .env.prod .env
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
echo "✅ Production environment started!"
echo "🌐 API: http://localhost:2005"
