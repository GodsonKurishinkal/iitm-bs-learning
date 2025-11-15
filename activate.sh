#!/bin/bash
# Activation script for IIT Madras BS Learning Environment

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}IIT Madras BS Data Science Learning Environment${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated (.venv)${NC}"
    echo ""
    echo "Python version: $(python --version)"
    echo "Pip version: $(pip --version)"
    echo ""
    echo -e "${GREEN}Ready to learn! 🚀${NC}"
    echo ""
    echo "Quick commands:"
    echo "  jupyter lab          - Start JupyterLab"
    echo "  jupyter notebook     - Start Jupyter Notebook"
    echo "  python script.py     - Run Python scripts"
    echo "  deactivate          - Exit virtual environment"
    echo ""
else
    echo "Error: .venv directory not found!"
    echo "Run: python3 -m venv .venv"
    exit 1
fi
