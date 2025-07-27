# FLNG Operations Analytics Platform

##  Overview

FLNG Operations Analytics Platform is a comprehensive Python-based solution for analyzing optimal weather windows for LNG tanker mooring operations at FLNG platforms.

##  Features

- Real-time weather data integration
- Safety score calculations based on OCIMF standards
- Interactive visualizations and heatmaps
- Vessel traffic monitoring (AIS)
- Multi-region support for Japanese waters

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/flng-operations-analytics.git
cd flng-operations-analytics

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1 # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run src/main.py

# Configuration

# Copy config/api_keys.example.yml to config/api_keys.yml
# Add your API keys (optional for basic functionality)
# Adjust safety thresholds in config/safety_thresholds.yml

#  Usage
# Access the application at http://localhost:8501
#  License
# MIT License
