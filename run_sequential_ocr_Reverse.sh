#!/bin/bash
# Sequential OCR Processor
# Processes one folder at a time, in frame numeric order
# Only updates Airtable after OCR and LLM processing is complete

set -e

# Load environment variables
source .env

# Log function
log() {
  echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Check for required commands
check_dependencies() {
  log "Checking dependencies..."
  
  if ! command -v node &> /dev/null; then
    log "ERROR: Node.js is required but not installed."
    exit 1
  fi
  
  if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
  log "ERROR: Python is required but not installed."
    exit 1
  fi
  
  # Check for required Node.js packages
  if [ ! -d "node_modules/airtable" ]; then
    log "Installing required npm packages..."
    npm install airtable dotenv
  fi
  
  # Check for required Python packages
  if ! python -c "import google.generativeai" 2>/dev/null && ! python3 -c "import google.generativeai" 2>/dev/null; then
    log "Installing required Python packages..."
    pip install google-generativeai pillow python-dotenv || pip3 install google-generativeai pillow python-dotenv
  fi
  
  log "All dependencies are installed."
}

# Display usage information
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --folder=PATH       Process a specific folder only"
  echo "  --model=NAME        Specify Gemini model to use (default: models/gemini-2.0-flash-exp)"
  echo "  --help              Display this usage information"
  echo ""
  echo "Example:"
  echo "  $0                  Process all folders sequentially with default model"
  echo "  $0 --folder=/path/to/folder --model=models/gemini-1.5-flash"
}

# Parse command line arguments
FOLDER=""
MODEL=""

for arg in "$@"; do
  case $arg in
    --folder=*)
      FOLDER="${arg#*=}"
      shift
      ;;
    --model=*)
      MODEL="${arg#*=}"
      shift
      ;;
    --ocr-only)
      echo "WARNING: --ocr-only option is no longer supported. Full LLM processing will be used."
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      log "Unknown option: $arg"
      usage
      exit 1
      ;;
  esac
done

# Configure API keys and models
setup_environment() {
  log "Setting up environment..."
  
  # Set up Gemini API key
  if [ -n "$GOOGLE_API_KEY" ]; then
    export GEMINI_API_KEY="$GOOGLE_API_KEY"
    log "Using Google API key as Gemini API key"
  elif [ -n "$GEMINI_API_KEY" ]; then
    log "Using existing Gemini API key"
  else
    log "WARNING: No API key found. Will attempt to proceed with LLM processing."
  fi
  
  # Set up preferred model
  if [ -n "$MODEL" ]; then
    export GEMINI_PREFERRED_MODEL="$MODEL"
    log "Using specified model: $MODEL"
  elif [ -n "$GEMINI_PREFERRED_MODEL" ]; then
    log "Using environment model: $GEMINI_PREFERRED_MODEL"
  else
    export GEMINI_PREFERRED_MODEL="models/gemini-2.0-flash-exp"
    log "Using default model: $GEMINI_PREFERRED_MODEL"
  fi
  
  # Set up fallback models if not already configured
  if [ -z "$GEMINI_FALLBACK_MODELS" ]; then
    export GEMINI_FALLBACK_MODELS="models/gemini-2.0-flash,models/gemini-1.5-flash"
    log "Using default fallback models: $GEMINI_FALLBACK_MODELS"
  fi
  
  # Note about OCR-only mode being disabled
  if [ -n "$OCR_ONLY_MODE" ]; then
    log "NOTE: OCR_ONLY_MODE environment variable is ignored. Full LLM processing will be used."
    unset OCR_ONLY_MODE
  fi
}

# Main execution
main() {
  log "Starting Sequential OCR Processor_Reverse..."
  
  # Check dependencies
  check_dependencies
  
  # Setup environment variables
  setup_environment
  
  # Create logs directory if it doesn't exist
  mkdir -p logs
  
  # Run the Sequential OCR Processor
  LOG_FILE="logs/sequential_ocr_$(date +%Y%m%d_%H%M%S).log"
  
  # Prepare command arguments
  CMD_ARGS=""
  
  if [ -n "$FOLDER" ]; then
    CMD_ARGS="--folder=\"$FOLDER\""
    log "Processing specific folder: $FOLDER"
  else
    log "Processing all folders chronologically, one at a time"
  fi
  
  # Add model parameter if set
  if [ -n "$GEMINI_PREFERRED_MODEL" ]; then
    CMD_ARGS="$CMD_ARGS --model=\"$GEMINI_PREFERRED_MODEL\""
  fi
  
  # Run node script with configured parameters
  if [ -n "$CMD_ARGS" ]; then
    node sequential_ocr_processor_Reverse.js $CMD_ARGS 2>&1 | tee "$LOG_FILE"
  else
    node sequential_ocr_processor_Reverse.js 2>&1 | tee "$LOG_FILE"
  fi
  
  if [ $? -eq 0 ]; then
    log "OCR processing completed successfully!"
    log "Log file: $LOG_FILE"
  else
    log "ERROR: OCR processing failed."
    log "Check log file for details: $LOG_FILE"
    exit 1
  fi
}

# Stop any previous instances
pkill -f "node sequential_ocr_processor_Reverse.js" || true

# Run the main function
main 