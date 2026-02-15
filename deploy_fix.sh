#!/bin/bash

# Speaker Separation Fix Deployment Script
# This script automates the process of updating and redeploying the fixed code

set -e  # Exit on error

echo "=================================================="
echo "Speaker Separation API - Deployment Fix Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed${NC}"
    echo "Please install docker-compose first"
    exit 1
fi

# Check if fixed file exists
if [ ! -f "improved_speaker_separator_fixed.py" ]; then
    echo -e "${RED}Error: improved_speaker_separator_fixed.py not found${NC}"
    echo "Please download the fixed file first"
    exit 1
fi

echo -e "${YELLOW}Step 1: Backing up current file...${NC}"
if [ -f "improved_speaker_separator.py" ]; then
    cp improved_speaker_separator.py improved_speaker_separator.py.backup
    echo -e "${GREEN}✓ Backup created: improved_speaker_separator.py.backup${NC}"
else
    echo -e "${YELLOW}No existing file to backup${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2: Replacing with fixed file...${NC}"
cp improved_speaker_separator_fixed.py improved_speaker_separator.py
echo -e "${GREEN}✓ File replaced${NC}"

echo ""
echo -e "${YELLOW}Step 3: Stopping current container...${NC}"
docker-compose down
echo -e "${GREEN}✓ Container stopped${NC}"

echo ""
echo -e "${YELLOW}Step 4: Rebuilding Docker image (this may take a few minutes)...${NC}"
docker-compose build --no-cache
echo -e "${GREEN}✓ Image rebuilt${NC}"

echo ""
echo -e "${YELLOW}Step 5: Starting container...${NC}"
docker-compose up -d
echo -e "${GREEN}✓ Container started${NC}"

echo ""
echo -e "${YELLOW}Step 6: Waiting for container to be ready...${NC}"
sleep 5

# Check if container is running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✓ Container is running${NC}"
else
    echo -e "${RED}✗ Container failed to start${NC}"
    echo "Check logs with: docker-compose logs speaker-api"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 7: Testing API health...${NC}"
sleep 3
if curl -f http://localhost:8000/health &> /dev/null; then
    echo -e "${GREEN}✓ API is healthy${NC}"
else
    echo -e "${RED}✗ API health check failed${NC}"
    echo "Check logs with: docker-compose logs speaker-api"
    exit 1
fi

echo ""
echo "=================================================="
echo -e "${GREEN}Deployment Complete!${NC}"
echo "=================================================="
echo ""
echo "📊 Status:"
echo "  • Container: Running"
echo "  • API: Healthy"
echo "  • UI: http://localhost:8000/ui"
echo ""
echo "🔍 Next Steps:"
echo "  1. Open http://localhost:8000/ui in your browser"
echo "  2. Upload your audio file"
echo "  3. The separation should now work correctly"
echo ""
echo "📝 Useful Commands:"
echo "  • View logs: docker-compose logs -f speaker-api"
echo "  • Restart: docker-compose restart"
echo "  • Stop: docker-compose down"
echo ""
echo "💾 Backup: improved_speaker_separator.py.backup"
echo ""
