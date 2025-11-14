#!/bin/bash
# Helper script for rebuilding Docker images with code updates

set -e

echo "🔄 Rebuilding VibeVoice with fresh code from GitHub..."
echo ""

# Generate cache bust value (timestamp)
CACHE_BUST=$(date +%Y%m%d-%H%M%S)
export CACHE_BUST

echo "📦 Cache bust value: $CACHE_BUST"
echo ""

# Rebuild backend-builder (includes git clone + frontend build + python venv)
echo "🔨 Step 1/2: Building backend-builder stage..."
docker compose build backend-builder

echo ""
echo "🔨 Step 2/2: Building final vibevoice image..."
docker compose build vibevoice

echo ""
echo "✅ Build complete! The following stages were rebuilt:"
echo "   - Stage 2: source-and-frontend (git clone + npm build)"
echo "   - Stage 3: python-builder (venv + pip install)"
echo "   - Stage 4: final image (assembly)"
echo ""
echo "💾 The following stages were cached:"
echo "   - Stage 1: model-downloader (3-4GB models)"
echo ""
echo "🚀 Start the container with: docker compose up -d vibevoice"
