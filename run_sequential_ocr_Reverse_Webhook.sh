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
    echo "  --folder=PATH Process a specific folder only"
    echo "  --help Display this usage information"
    echo ""
    echo "Example:"
    echo "  $0 Process all folders sequentially"
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
    
    # Run the command and capture output to log file
    if [ -n "$FOLDER" ]; then
        log "Processing specific folder: $FOLDER"
        node sequential_ocr_processor_Reverse.js --folder="$FOLDER" 2>&1 | tee "$LOG_FILE"
    else
        log "Processing all folders chronologically, one at a time"
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