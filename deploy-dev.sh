#!/bin/bash
echo "🚀 Starting Development Environment..."
cp .env.dev .env
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
echo "✅ Development environment started!"
echo "🌐 API: http://localhost:2005"
echo "📖 Docs: http://localhost:2005/docs"