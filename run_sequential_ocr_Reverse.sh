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
  
  log "All dependencies are installed."
}

# Display usage information
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  --folder=PATH     Process a specific folder only"
  echo "  --help            Display this usage information"
  echo ""
  echo "Example:"
  echo "  $0                Process all folders sequentially"
  echo "  $0 --folder=/home/jason/Videos/screenRecordings/screen_recording_2025_04_02_at_6_07_21_pm"
}

# Parse command line arguments
FOLDER=""

for arg in "$@"; do
  case $arg in
    --folder=*)
      FOLDER="${arg#*=}"
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

# Main execution
main() {
  log "Starting Sequential OCR Processor_Reverse..."
  
  # Check dependencies
  check_dependencies
  
  # Create logs directory if it doesn't exist
  mkdir -p logs
  
  # Run the Sequential OCR Processor
  LOG_FILE="logs/sequential_ocr_$(date +%Y%m%d_%H%M%S).log"
  
  if [ -n "$FOLDER" ]; then
    log "Processing specific folder: $FOLDER"
    node sequential_ocr_processor_Reverse.js --folder="$FOLDER" 2>&1 | tee "$LOG_FILE"
  else
    log "Processing all folders chronologically, one at a time"
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