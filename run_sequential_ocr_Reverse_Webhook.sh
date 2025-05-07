#!/bin/bash
# Sequential OCR Processor
# Processes one folder at a time, in frame numeric order
# Only updates Airtable after OCR and LLM processing is complete

set -e

# Color definitions for console output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration variables
MAX_LLM_RETRIES=3
RETRY_DELAY_BASE=1.0
RETRY_DELAY_MAX=5.0

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
    
    # Run the command and capture output to log file
    if [ -n "$CMD_ARGS" ]; then
        node sequential_ocr_processor_Reverse.js $CMD_ARGS 2>&1 | tee "$LOG_FILE"
    else
        node sequential_ocr_processor_Reverse.js 2>&1 | tee "$LOG_FILE"
    fi
    
    # Save the exit code
    EXIT_CODE=$?
    
    # Log final status
    if [ $EXIT_CODE -eq 0 ]; then
        log "OCR processing completed successfully!"
        log "Log file: $LOG_FILE"
        STATUS="success"
    else
        log "ERROR: OCR processing failed."
        log "Check log file for details: $LOG_FILE"
        STATUS="error"
    fi
    
    # Send webhook with the last 3 lines as payload
    TEMP_SCRIPT=$(mktemp)
    
    cat > "$TEMP_SCRIPT" << EOF
const fs = require('fs');
const http = require('http');

// Define webhook URLs
const FIRST_WEBHOOK_URL = '/webhook/5e74dbc0-c46e-41e5-897b-78df0e9813ce';
const SECOND_WEBHOOK_URL = '/webhook/7a44da2c-6f7b-4a0a-979b-1f208ab3f969';

// Read the last 3 lines from the log file
const logFile = '$LOG_FILE';
const content = fs.readFileSync(logFile, 'utf8');
const lines = content.split('\n').filter(line => line.trim() !== '');
const last3Lines = lines.slice(-3).join('\n');

// Prepare the webhook data
const data = JSON.stringify({
  status: '$STATUS',
  summary: last3Lines
});

// Function to send webhook
function sendWebhook(webhookPath) {
  return new Promise((resolve, reject) => {
    const req = http.request({
      hostname: 'localhost',
      port: 5678,
      path: webhookPath,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length
      }
    }, (res) => {
      console.log(\`Webhook sent to \${webhookPath} with status code: \${res.statusCode}\`);
      resolve();
    });

    req.on('error', (e) => {
      console.error(\`Error sending webhook to \${webhookPath}: \${e.message}\`);
      reject(e);
    });

    req.write(data);
    req.end();
  });
}

// Execute webhooks with delay
async function executeWebhooks() {
  try {
    // Send to first webhook
    console.log('Sending to first webhook...');
    await sendWebhook(FIRST_WEBHOOK_URL);
    
    // Wait for 1 minute (60000 milliseconds)
    console.log('Waiting 1 minute before sending to second webhook...');
    await new Promise(resolve => setTimeout(resolve, 60000));
    
    // Send to second webhook
    console.log('Sending to second webhook...');
    await sendWebhook(SECOND_WEBHOOK_URL);
    
    console.log('All webhooks sent successfully');
  } catch (error) {
    console.error('Error in webhook sequence:', error);
  }
}

// Start the webhook sequence
executeWebhooks();
EOF
    
    # Execute the node script to send the webhook
    log "Sending webhooks with 1-minute delay between them..."
    node "$TEMP_SCRIPT"
    
    # Clean up
    rm -f "$TEMP_SCRIPT"
    
    # Return with the original exit code
    exit $EXIT_CODE
}

# Stop any previous instances
pkill -f "node sequential_ocr_processor_Reverse.js" || true

# Run the main function
main