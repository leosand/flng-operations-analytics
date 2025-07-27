#!/bin/bash

# FLNG Operations Analytics Platform Launcher

echo " FLNG Operations Analytics Platform"
echo "====================================="

# Check Python installation
if ! command -v python3 &> /dev/null; then
   echo " Python not found. Please install Python 3.9+"
   exit 1
fi

echo " Python found: $(python3 --version)"

# Navigate to script directory
cd "$(dirname "$0")"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
   echo "Creating virtual environment..."
   python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Create directories
for dir in "data/cache" "logs" "data/exports"; do
   mkdir -p "$dir"
done

# Check config
if [ ! -f "config/api_keys.yml" ]; then
   if [ -f "config/api_keys.example.yml" ]; then
       cp config/api_keys.example.yml config/api_keys.yml
       echo "Created config/api_keys.yml from example"
   fi
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export STREAMLIT_THEME_BASE=dark

# Launch application
echo ""
echo " Starting application..."
echo " Access at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"

streamlit run src/main.py --server.port=8501 --server.address=0.0.0.0
