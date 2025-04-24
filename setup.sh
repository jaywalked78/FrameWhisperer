#!/bin/bash
# Setup script for OCRPipeline
# Creates necessary directories and copies .env configuration

set -e  # Exit on any error

echo "Setting up OCRPipeline environment..."

# Create necessary directories
mkdir -p logs/ocr
mkdir -p output/ocr_results
mkdir -p output/json

# Create .gitkeep files to preserve directory structure in git
touch logs/.gitkeep
touch output/.gitkeep
touch output/ocr_results/.gitkeep
touch output/json/.gitkeep

# Check for .env.ocr in parent directory
if [ -f "../.env.ocr" ]; then
    echo "Found .env.ocr in parent directory, copying to .env..."
    cp "../.env.ocr" ./.env
    echo "Copied .env.ocr to .env"
    
    # Check if we need to add Gemini API keys
    if ! grep -q "GOOGLE_API_KEY" ./.env; then
        echo "Adding GOOGLE_API_KEY placeholder to .env..."
        echo -e "\n# Google Gemini API Keys (Required for LLM processing)" >> ./.env
        echo "GOOGLE_API_KEY=your_primary_gemini_api_key  # Add your Gemini API key here" >> ./.env
        echo "Added GOOGLE_API_KEY placeholder to .env (you'll need to add your actual API key)"
    fi
else
    echo ".env.ocr not found in parent directory."
    echo "Creating .env from .env.example template..."
    cp ./.env.example ./.env
    echo "Created .env from template. Please edit with your actual credentials."
fi

# Make scripts executable
chmod +x run_ocr_pipeline.sh
chmod +x run_sequential_ocr_Reverse.sh
chmod +x run_sequential_ocr_Reverse_Webhook.sh

echo "Setup complete! Next steps:"
echo "1. Edit .env file with your API keys and configuration settings"
echo "2. Install Python dependencies: pip install -r requirements.txt"
echo "3. Install Node.js dependencies: npm install"
echo "4. Run the pipeline with: ./run_ocr_pipeline.sh" 