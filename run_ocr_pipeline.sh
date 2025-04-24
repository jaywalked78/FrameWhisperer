#!/bin/bash
# OCR Pipeline - Main Runner
# Combines capabilities of various OCR processing scripts

set -e

# Load environment variables
source .env

# Log function
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1"
}

# Display usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --mode=MODE         Processing mode (sequential, reverse, parallel)"
    echo "  --folder=PATH       Process a specific folder only"
    echo "  --skip-airtable     Skip updating Airtable (JSON output only)"
    echo "  --webhook           Send webhook notifications on completion"
    echo "  --help              Display this usage information"
    echo ""
    echo "Modes:"
    echo "  sequential          Process frames one at a time in chronological order"
    echo "  reverse             Process frames one at a time in reverse chronological order"
    echo "  parallel            Process frames in parallel with multiple workers"
    echo ""
    echo "Example:"
    echo "  $0 --mode=sequential                       Process all folders sequentially"
    echo "  $0 --mode=reverse --folder=/path/to/folder Process specific folder in reverse order"
    echo "  $0 --mode=parallel --webhook               Parallel processing with webhook notifications"
}

# Create logs directory if it doesn't exist
mkdir -p logs/ocr
mkdir -p output/ocr_results

# Initialize variables with defaults
MODE="sequential"
FOLDER=""
SKIP_AIRTABLE=false
WEBHOOK=false

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --mode=*)
            MODE="${arg#*=}"
            shift
        ;;
        --folder=*)
            FOLDER="${arg#*=}"
            shift
        ;;
        --skip-airtable)
            SKIP_AIRTABLE=true
            shift
        ;;
        --webhook)
            WEBHOOK=true
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

# Validate mode
if [[ "$MODE" != "sequential" && "$MODE" != "reverse" && "$MODE" != "parallel" ]]; then
    log "Error: Invalid mode '$MODE'. Must be one of: sequential, reverse, parallel"
    usage
    exit 1
fi

# Main execution
log "Starting OCR Pipeline in $MODE mode"

# Execute the appropriate script based on mode
case $MODE in
    sequential)
        if [ "$WEBHOOK" = true ]; then
            log "Running sequential processing with webhook notifications"
            if [ -n "$FOLDER" ]; then
                ./run_sequential_ocr_Reverse_Webhook.sh --folder="$FOLDER"
            else
                ./run_sequential_ocr_Reverse_Webhook.sh
            fi
        else
            log "Running sequential processing without webhook"
            if [ -n "$FOLDER" ]; then
                ./run_sequential_ocr_Reverse.sh --folder="$FOLDER"
            else
                ./run_sequential_ocr_Reverse.sh
            fi
        fi
        ;;
    reverse)
        log "Running reverse chronological processing"
        if [ -n "$FOLDER" ]; then
            node process_remaining_frames_reverse.js --folder="$FOLDER"
        else
            node process_remaining_frames_reverse.js
        fi
        ;;
    parallel)
        log "Running parallel processing"
        if [ -n "$FOLDER" ]; then
            node process_remaining_frames_sequential.js --folder="$FOLDER"
        else
            node process_remaining_frames_sequential.js
        fi
        ;;
esac

# Check if python processing should run
if [ "$SKIP_AIRTABLE" = true ]; then
    SKIP_ARG="--skip-airtable-update"
else
    SKIP_ARG=""
fi

# If specific folder is provided, also run Python processor directly
if [ -n "$FOLDER" ]; then
    log "Running Python OCR processor on folder: $FOLDER"
    python process_frames_by_path_Reverse.py --folder-path "$FOLDER" $SKIP_ARG
fi

log "OCR Pipeline processing completed" 