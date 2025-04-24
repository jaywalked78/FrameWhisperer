# OCR Pipeline

A comprehensive system for processing video frames with OCR and LLM analysis, designed to extract and analyze text from screenshots and images. This pipeline integrates seamlessly with Airtable for data management and utilizes Google's Gemini API for advanced text processing.

## Architecture

The OCR Pipeline consists of several key components:

```mermaid
flowchart TD
    A[Video Frames] --> B["OCR Processing (process_frames_by_path.py)"]
    B --> C["LLM Analysis (Gemini API)"]
    C --> D[(Airtable)]
    E[Screen Recordings] --> F["Sequential Processing (process_remaining_frames_sequential.js)"]
    F --> B
    G[Screen Recordings] --> H["Reverse Order Processing (process_remaining_frames_reverse.js)"]
    H --> B
```

## Key Components

### Main Processors

- **process_frames_by_path.py**: Core Python script that handles OCR text extraction and LLM analysis for individual frames
- **process_frames_by_path_Reverse.py**: Variant that processes frames in reverse order
- **process_remaining_frames_sequential.js**: JavaScript processor for sequential frame processing
- **process_remaining_frames_reverse.js**: JavaScript processor for reverse-order frame processing
- **robust_ocr_worker.js**: JavaScript worker for robust OCR processing with retries and error handling
- **sequential_ocr_processor_Reverse.js**: Orchestrator for sequential OCR processing

### Runner Scripts

- **run_ocr_pipeline.sh**: Main unified script that provides options for different processing modes
- **run_sequential_ocr_Reverse_Webhook.sh**: Shell script to run the OCR pipeline with webhook notifications
- **run_sequential_ocr_Reverse.sh**: Shell script for running sequential OCR without webhooks

## Features

- **Intelligent Text Extraction**: Uses Tesseract OCR for initial text extraction
- **LLM Enhancement**: Utilizes Google's Gemini AI to clean OCR results and extract meaningful text
- **Sensitive Information Detection**: Automatically flags frames containing sensitive information:
  - API keys (Google, AWS, Stripe, etc.)
  - Passwords and credentials
  - Payment card numbers (validated with Luhn algorithm)
  - Environment variables with sensitive values
- **Airtable Integration**: Stores processed results in Airtable for integration with other systems:
  - Structured OCR data in designated fields
  - Flagged status for frames with sensitive content
  - Detailed explanations of detected sensitive information
- **Webhook Notifications**: Notifies external systems (like n8n) when processing is complete
- **API Key Rotation**: Supports rotation of API keys to avoid rate limits
- **Error Handling**: Robust retry logic and error handling for maximum reliability

## Processing Options

### Sequential Processing
- Processes one frame at a time in chronological order
- Ideal for ensuring consistent, reliable processing
- Updates Airtable after each frame is processed

### Reverse Processing
- Processes frames in reverse chronological order (newest first)
- Useful for prioritizing recent frames

### Parallel Processing
- Processes multiple frames concurrently with worker processes
- Significantly faster for large batches of frames
- Includes API key rotation to avoid rate limits

## Usage

### Prerequisites

- Python 3.8+
- Node.js 14+
- Tesseract OCR installed (`apt-get install tesseract-ocr`)
- Google Gemini API key(s)
- Airtable Personal Access Token and Base ID

### Setup

1. Configure the `.env` file with your API keys and settings (copy from `.env.example`)
2. Install Python dependencies: `pip install -r requirements.txt`
3. Install Node.js dependencies: `npm install`

### Running the Pipeline

#### Unified Runner Script

The `run_ocr_pipeline.sh` script provides a unified interface for all processing options:

```bash
# Sequential processing (default)
./run_ocr_pipeline.sh --mode=sequential

# Process a specific folder in reverse order
./run_ocr_pipeline.sh --mode=reverse --folder=/path/to/folder

# Parallel processing with webhook notifications
./run_ocr_pipeline.sh --mode=parallel --webhook

# Skip Airtable updates (JSON output only)
./run_ocr_pipeline.sh --mode=sequential --skip-airtable
```

#### Specific Processor Scripts

For sequential processing with webhook notifications:
```bash
./run_sequential_ocr_Reverse_Webhook.sh
```

For processing a specific folder:
```bash
./run_sequential_ocr_Reverse_Webhook.sh --folder=/path/to/folder
```

For running the JavaScript processors directly:
```bash
# Sequential processing
node process_remaining_frames_sequential.js

# Reverse order processing
node process_remaining_frames_reverse.js
```

## Airtable Integration

The system integrates with Airtable to store OCR results and metadata:

### Field Structure
- **OCRData**: Cleaned and structured text extracted from frame images
- **Flagged**: Boolean flag indicating if sensitive content was detected
- **SensitivityConcerns**: Detailed explanation of the sensitive information detected

### Update Process
- Records are updated only after successful OCR and LLM processing
- Batch updates respect Airtable's 10-record batch update limit
- Rate limiting prevents hitting Airtable API limits

## Environment Variables

Key environment variables needed in the `.env` file:

- `AIRTABLE_PERSONAL_ACCESS_TOKEN`: Your Airtable PAT
- `AIRTABLE_BASE_ID`: Your Airtable base ID
- `AIRTABLE_TABLE_NAME`: Table name for frame data (default: tblFrameAnalysis)
- `AIRTABLE_TRACKING_TABLE`: Table for tracking processed folders (default: Finished OCR Processed Folders)
- `AIRTABLE_RATE_LIMIT_SLEEP`: Sleep time between Airtable API calls (default: 0.25s)
- `GOOGLE_API_KEY`: Gemini API key
- `GEMINI_API_KEY_1` through `GEMINI_API_KEY_5`: Additional Gemini API keys for rotation
- `BASE_DIR`: Base directory for screen recordings
- `MAX_WORKERS`: Maximum number of concurrent workers for parallel processing 