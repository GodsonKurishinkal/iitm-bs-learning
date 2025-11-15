#!/bin/bash
# Activation script for IIT Madras BS Learning Environment

echo "🎓 Activating IIT Madras BS Learning Environment..."
source venv/bin/activate
echo "✅ Virtual environment activated!"
echo ""
echo "📦 Installed packages:"
pip list --format=columns | head -20
echo ""
echo "💡 Tips:"
echo "   - To deactivate: deactivate"
echo "   - To install packages: pip install <package>"
echo "   - To update requirements: pip freeze > requirements.txt"
echo ""
