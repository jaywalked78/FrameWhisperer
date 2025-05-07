#!/usr/bin/env python3
"""
Process Frames By Path - OCR and LLM Processor

This script searches for Airtable records matching a specific FolderPath pattern,
processes the images with OCR, sends the text to an LLM for analysis,
and updates the original Airtable records with the results.

This workflow is designed to be triggered by n8n, using:
({FolderPath} = '/home/jason/Videos/screenRecordings/{{ $('GetFolderName').first().json.name }}/{{ $('GetFrameName').first().json.name }}')

Usage:
  python process_frames_by_path.py --folder-path "/path/to/folder" --frame-pattern "frame_*.jpg"
  python process_frames_by_path.py --folder-path-pattern "/path/to/folder/frame_*.jpg"
  python process_frames_by_path.py --batch-size 20 --limit 100
  python process_frames_by_path.py --specific-ids frame_ids.txt --folder-path-pattern "/path/to/folder/*.jpg"
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import datetime
import time
import requests
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
import pytesseract
from tqdm import tqdm
import aiofiles

# Import the GeminiProcessor if available, otherwise use direct API calls
try:
    from gemini_processor import GeminiProcessor, ProcessingResult
    GEMINI_PROCESSOR_AVAILABLE = True
    print("Using GeminiProcessor for advanced model selection and fallbacks")
except ImportError:
    GEMINI_PROCESSOR_AVAILABLE = False
    print("GeminiProcessor not found, using direct API calls")

# Configure logging
os.makedirs("logs/ocr", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/ocr/process_frames_by_path_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("process_frames_by_path")

# Load environment variables
load_dotenv()
AIRTABLE_TOKEN = os.environ.get('AIRTABLE_PERSONAL_ACCESS_TOKEN')
AIRTABLE_BASE_ID = os.environ.get('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = os.environ.get('AIRTABLE_TABLE_NAME', 'tblFrameAnalysis')

# Global flag to control OCR-only mode (no LLM)
# Modified to always be False - OCR Only mode disabled
OCR_ONLY_MODE = False

# Global flag to track if API validation was already successful
API_VALIDATION_SUCCESSFUL = False

# Get preferred model from environment variables
GEMINI_PREFERRED_MODEL = os.environ.get('GEMINI_PREFERRED_MODEL', 'models/gemini-2.0-flash-exp')
GEMINI_FALLBACK_MODELS = os.environ.get('GEMINI_FALLBACK_MODELS', 'models/gemini-2.0-flash,models/gemini-1.5-flash').split(',')

# Function to validate a Gemini API key
def validate_gemini_key(api_key):
    """Test if a Gemini API key is valid by trying different model versions."""
    # Check if validation was already successful
    global API_VALIDATION_SUCCESSFUL
    if API_VALIDATION_SUCCESSFUL:
        logger.info("Skipping API key validation as it was previously successful")
        return True
    
    if not api_key or len(api_key) < 10:
        logger.error("API key is empty or too short to be valid")
        return False
        
    # Check if the API key matches the pattern of a valid Google API key
    import re
    if not re.match(r'^AIza[0-9A-Za-z_-]{35}$', api_key):
        logger.error("API key doesn't match expected Google API key format")
        return False
    
    # If GeminiProcessor is available, use it for validation
    if GEMINI_PROCESSOR_AVAILABLE:
        try:
            processor = GeminiProcessor(api_key=api_key, max_retries=1, allow_experimental=True)
            models = processor.list_available_models(force_refresh=True)
            model_count = len(models)
            logger.info(f"✓ API key validation SUCCESSFUL using GeminiProcessor. Found {model_count} models.")
            API_VALIDATION_SUCCESSFUL = True
            return True
        except Exception as e:
            logger.error(f"❌ GeminiProcessor validation failed: {str(e)}")
            # Fall back to direct API testing
    
    # Direct API testing if GeminiProcessor failed or is not available
    # Models to try in order of preference
    models_to_try = ['gemini-1.5-flash', 'gemini-pro']
    
    for model_name in models_to_try:
        try:
            logger.info(f"Validating API key with model: {model_name}")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Test")
            logger.info(f"✓ API key validation SUCCESSFUL with model: {model_name}")
            API_VALIDATION_SUCCESSFUL = True
            return True
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"API key validation failed with model {model_name}: {error_msg}")
            
            # Check for specific error conditions
            if "API_KEY_INVALID" in error_msg or "key expired" in error_msg:
                logger.error(f"❌ API key is invalid or expired. Please update your API key.")
                return False
            
            if "PERMISSION_DENIED" in error_msg:
                logger.error(f"❌ API key doesn't have permission to access model {model_name}")
                # Continue and try another model
                continue
                
            if "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                logger.error(f"❌ API quota exceeded. Either wait or get a new API key.")
                return False
    
    # If we get here, all models failed
    logger.error("❌ API key validation failed with all model variants. Please check your API key.")
    return False

# Get Gemini API key from .env
GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')

# Log the API key (partially masked for security)
if GEMINI_API_KEY:
    masked_key = GEMINI_API_KEY[:6] + "..." + GEMINI_API_KEY[-4:]
    logger.info(f"Found API key in environment: {masked_key}")
else:
    logger.error("No API key found in environment variables. OCR processing will fail.")
    GEMINI_API_KEY = "" # Empty string that will cause validate_gemini_key to fail

# Reload .env file to ensure we have the latest key
try:
    logger.info("Attempting to reload .env file to ensure latest API key")
    load_dotenv(override=True)
    fresh_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
    if fresh_key != GEMINI_API_KEY and fresh_key:
        logger.info(f"Updated API key found after reload! Using new key.")
        GEMINI_API_KEY = fresh_key
        masked_key = GEMINI_API_KEY[:6] + "..." + GEMINI_API_KEY[-4:] if GEMINI_API_KEY else "None"
        logger.info(f"Now using API key: {masked_key}")
except Exception as env_err:
    logger.error(f"Error reloading .env file: {str(env_err)}")

logger.info(f"Using API key: {GEMINI_API_KEY[:6] + '...' + GEMINI_API_KEY[-4:] if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10 else 'None or invalid key'}")
logger.info(f"Preferred model: {GEMINI_PREFERRED_MODEL}")
logger.info(f"Fallback models: {GEMINI_FALLBACK_MODELS}")

# Initialize GeminiProcessor if available
gemini_processor = None
if GEMINI_PROCESSOR_AVAILABLE and GEMINI_API_KEY:
    try:
        gemini_processor = GeminiProcessor(
            api_key=GEMINI_API_KEY,
            default_model=GEMINI_PREFERRED_MODEL,
            allow_experimental=True,
            max_retries=2
        )
        logger.info(f"Initialized GeminiProcessor with model: {GEMINI_PREFERRED_MODEL}")
    except Exception as init_err:
        logger.error(f"Failed to initialize GeminiProcessor: {str(init_err)}")
        logger.warning("⚠️ Will attempt to use direct API calls instead.")
        gemini_processor = None
elif not GEMINI_PROCESSOR_AVAILABLE:
    logger.warning("GeminiProcessor module not available. Using direct API calls.")
else:
    logger.warning("No API key available. Direct API calls will be attempted.")

# Configure Gemini for direct API calls
if GEMINI_API_KEY and not gemini_processor:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info(f"Configured Gemini with API key")
    except Exception as config_err:
        logger.error(f"Error configuring Gemini API: {str(config_err)}")
        OCR_ONLY_MODE = True
    
# Validate the API key
if not OCR_ONLY_MODE:
    is_valid = validate_gemini_key(GEMINI_API_KEY)
    if not is_valid:
        logger.error("Invalid or missing Gemini API key. Processing may fail.")
        logger.warning("⚠️ Invalid API key, but continuing with LLM processing as OCR-ONLY mode is disabled.")
    else:
        logger.info("✓ API key validation passed. Full OCR+LLM processing will be performed.")
        
        # Even if OCR_ONLY_MODE is explicitly set in environment, we ignore it
        if os.environ.get('OCR_ONLY_MODE', '').lower() in ('true', '1', 'yes'):
            logger.info("OCR_ONLY_MODE found in environment but ignored as per configuration.")
else:
    logger.info("OCR_ONLY_MODE is forced to False. LLM processing will always be attempted.")

# Simplify for now - no key rotation
GEMINI_USE_KEY_ROTATION = False

# Define output directory for OCR results
OCR_RESULTS_DIR = "output/ocr_results"
os.makedirs(OCR_RESULTS_DIR, exist_ok=True)

class AirtableConnector:
    """
    Class to interact with Airtable API.
    """
    
    def __init__(self, token, base_id, table_name):
        """
        Initialize the Airtable connector.
        
        Args:
            token: Airtable API token
            base_id: Airtable base ID
            table_name: Airtable table name
        """
        self.token = token
        self.base_id = base_id
        self.table_name = table_name
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.base_url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
        
        # Rate limiting to avoid hitting Airtable API limits
        self.rate_limit_sleep = float(os.environ.get('AIRTABLE_RATE_LIMIT_SLEEP', '0.25'))
        
        logger.info(f"Initialized Airtable connector for {table_name} in base {base_id}")
    
    async def find_records_by_folder_path(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Find all Airtable records matching a specific FolderPath.
        
        Args:
            folder_path: Path to search for
            
        Returns:
            List of matching Airtable records
        """
        try:
            logger.info(f"Finding Airtable records for path: {folder_path}")
            
            # Escape single quotes in the path for the formula
            safe_path = folder_path.replace("'", "\\'")
            formula = f"{{FolderPath}}='{safe_path}'"
            
            # Apply rate limiting
            await asyncio.sleep(self.rate_limit_sleep)
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params={
                    "filterByFormula": formula,
                }
            )
            
            if response.status_code != 200:
                logger.error(f"Airtable API error: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            records = data.get('records', [])
            
            logger.info(f"Found {len(records)} Airtable records for path {folder_path}")
            
            return records
        except Exception as e:
            logger.error(f"Error finding records for path {folder_path}: {e}")
            return []
    
    async def find_records_by_path_pattern(self, path_pattern: str, limit: int = None) -> List[Dict[str, Any]]:
        """
        Find Airtable records where FolderPath matches a pattern.
        Used for batch processing multiple records.
        
        Args:
            path_pattern: Pattern to match in the FolderPath field
            limit: Optional limit on number of records to return
            
        Returns:
            List of matching Airtable records sorted from oldest to newest
        """
        try:
            logger.info(f"Finding Airtable records for path pattern: {path_pattern}")
            
            # If we have a full path with wildcards, just use the directory part
            if '*' in path_pattern:
                directory = os.path.dirname(path_pattern)
                # If the directory is just ., use a more general approach
                if directory == '.':
                    logger.info("Using general query for wildcards")
                    formula = f"AND(NOT({{OCRData}}), NOT({{FolderPath}} = ''))"
                else:
                    safe_dir = directory.replace("'", "\\'")
                    formula = f"FIND('{safe_dir}', {{FolderPath}}) > 0"
            else:
                # If no wildcards, use exact match
                safe_path = path_pattern.replace("'", "\\'")
                formula = f"{{FolderPath}}='{safe_path}'"
            
            # Apply rate limiting
            await asyncio.sleep(self.rate_limit_sleep)
            
            params = {
                "filterByFormula": formula,
                # Sort records from oldest to newest based on folder naming convention
                # which usually includes a date/timestamp
                "sort[0][field]": "FolderName",
                "sort[0][direction]": "asc"
            }
            
            if limit:
                params["maxRecords"] = limit
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=params
            )
            
            if response.status_code != 200:
                logger.error(f"Airtable API error: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            records = data.get('records', [])
            
            logger.info(f"Found {len(records)} Airtable records for path pattern {path_pattern}")
            
            # Secondary sort: ensure chronological order by parsing folder names/dates
            # This assumes folder names follow pattern like screen_recording_YYYY_MM_DD...
            # If FolderName doesn't perfectly sort, we'll use FolderPath to extract date info
            try:
                records.sort(key=lambda r: self._extract_date_from_path(r['fields'].get('FolderPath', '')))
                logger.info("Records sorted chronologically from oldest to newest")
            except Exception as sort_err:
                logger.warning(f"Error during chronological sorting: {sort_err}. Using default Airtable sort.")
            
            return records
        except Exception as e:
            logger.error(f"Error finding records for path pattern {path_pattern}: {e}")
            return []
    
    def _extract_date_from_path(self, path: str) -> str:
        """
        Extract date information from a folder path for chronological sorting.
        
        Args:
            path: File or folder path
            
        Returns:
            String representation of date for sorting (YYYY_MM_DD)
        """
        try:
            # Extract the folder name from the path
            if not path:
                return "0000_00_00"  # Default for empty paths
                
            folder_name = os.path.basename(os.path.dirname(path))
            
            # For paths like /home/user/screen_recording_2025_04_07_at_10_10_12_pm/frame.jpg
            if folder_name.startswith("screen_recording_"):
                # Extract YYYY_MM_DD portion
                parts = folder_name.split("_")
                if len(parts) >= 5:
                    return f"{parts[2]}_{parts[3]}_{parts[4]}"
            
            # Fallback: use the whole path for consistent (but not chronological) sorting
            return path
        except Exception:
            return path  # Fallback to using the raw path
    
    async def update_record(self, record_id: str, fields: Dict[str, Any]) -> bool:
        """
        Update an Airtable record.
        
        Args:
            record_id: Airtable record ID
            fields: Fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Updating Airtable record: {record_id}")
            
            # Apply rate limiting
            await asyncio.sleep(self.rate_limit_sleep)
            
            # Prepare the update payload
            update_data = {
                "fields": fields
            }
            
            response = requests.patch(
                f"{self.base_url}/{record_id}",
                headers=self.headers,
                json=update_data
            )
            
            if response.status_code != 200:
                logger.error(f"Airtable API error: {response.status_code} - {response.text}")
                return False
                
            logger.info(f"Successfully updated Airtable record: {record_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating Airtable record {record_id}: {e}")
            return False
            
    async def batch_update_records(self, updates: List[Dict[str, Any]]) -> bool:
        """
        Update multiple Airtable records in a single API call.
        Respects Airtable's 10 record per batch limit.
        
        Args:
            updates: List of updates, each with 'id' and 'fields' keys
            
        Returns:
            True if all updates were successful, False if any failed
        """
        try:
            total_records = len(updates)
            logger.info(f"Batch updating {total_records} Airtable records")
            
            # Process in batches of 10 (Airtable's limit)
            batch_size = 10
            all_success = True
            
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(total_records + batch_size - 1)//batch_size}")
                
                # Apply rate limiting
                await asyncio.sleep(self.rate_limit_sleep)
                
                # Prepare the update payload
                payload = {
                    "records": batch
                }
                
                response = requests.patch(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    logger.error(f"Airtable batch update error: {response.status_code} - {response.text}")
                    all_success = False
                else:
                    logger.info(f"Successfully updated batch of {len(batch)} Airtable records")
                
                # Additional rate limiting between batches
                await asyncio.sleep(self.rate_limit_sleep * 2)
            
            if all_success:
                logger.info(f"Successfully updated all {total_records} Airtable records")
            else:
                logger.warning(f"Some batches failed during update of {total_records} records")
                
            return all_success
        except Exception as e:
            logger.error(f"Error batch updating Airtable records: {e}")
            return False

    async def find_records_by_ids(self, record_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Find Airtable records by their IDs.
        
        Args:
            record_ids: List of record IDs to fetch
            
        Returns:
            List of matching Airtable records
        """
        try:
            logger.info(f"Finding Airtable records by IDs: {len(record_ids)} records")
            
            records = []
            
            # Process in batches of 10 to avoid rate limiting
            batch_size = 10
            for i in range(0, len(record_ids), batch_size):
                batch_ids = record_ids[i:i+batch_size]
                id_formula_parts = [f"RECORD_ID() = '{id}'" for id in batch_ids]
                formula = f"OR({','.join(id_formula_parts)})"
                
                # Apply rate limiting
                await asyncio.sleep(self.rate_limit_sleep)
                
                response = requests.get(
                    self.base_url,
                    headers=self.headers,
                    params={
                        "filterByFormula": formula,
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Airtable API error: {response.status_code} - {response.text}")
                    continue
                    
                data = response.json()
                batch_records = data.get('records', [])
                records.extend(batch_records)
                
                logger.info(f"Fetched batch of {len(batch_records)} records")
            
            logger.info(f"Found {len(records)} Airtable records from {len(record_ids)} IDs")
            return records
            
        except Exception as e:
            logger.error(f"Error finding records by IDs: {e}")
            return []


class ImageOCRProcessor:
    def __init__(self):
        """Initialize the OCR processor with credentials and model."""
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variables
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.error("No API key found in environment variables")
            logger.warning("⚠️ Running in OCR-ONLY mode (no LLM processing)")
            global OCR_ONLY_MODE
            OCR_ONLY_MODE = True
        else:
            # Use GeminiProcessor if available
            global gemini_processor
            if GEMINI_PROCESSOR_AVAILABLE and not gemini_processor:
                try:
                    logger.info("Initializing GeminiProcessor")
                    gemini_processor = GeminiProcessor(
                        api_key=api_key,
                        default_model=GEMINI_PREFERRED_MODEL,
                        allow_experimental=True
                    )
                    # Test if processor is working
                    models = gemini_processor.list_available_models()
                    logger.info(f"GeminiProcessor initialized with {len(models)} available models")
                except Exception as e:
                    logger.error(f"Failed to initialize GeminiProcessor: {str(e)}")
                    gemini_processor = None
                    
            # Configure direct Gemini API as fallback
            if not gemini_processor:
                try:
                    logger.info("Initializing direct Gemini API with provided key")
                    genai.configure(api_key=api_key)
                    # Set up the Gemini Pro Vision model
                    self.model = genai.GenerativeModel(GEMINI_PREFERRED_MODEL)
                    # Test API key validity with a simple request
                    self._validate_api_key()
                except Exception as e:
                    error_message = f"Failed to initialize Gemini API: {str(e)}"
                    logger.error(error_message)
                    logger.warning("⚠️ Running in OCR-ONLY mode (no LLM processing)")
                    global OCR_ONLY_MODE
                    OCR_ONLY_MODE = True
        
        # Configure pytesseract path if necessary (Windows)
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        logger.info(f"ImageOCRProcessor initialized. OCR-ONLY mode: {OCR_ONLY_MODE}")

    def _validate_api_key(self):
        """Validate the API key with a simple request."""
        try:
            # If using GeminiProcessor, it's already validated
            global gemini_processor
            if gemini_processor:
                logger.info("✅ API key already validated through GeminiProcessor")
                return True
                
            # Create a minimal text-only model to test API key
            test_model = genai.GenerativeModel('gemini-pro')
            # Make a simple request
            response = test_model.generate_content("Hello, this is a test request to validate API key.")
            logger.info("✅ API key validation successful")
            return True
        except Exception as e:
            error_message = f"API key validation failed: {str(e)}"
            logger.error(error_message)
            # Set global flag for OCR-only mode
            global OCR_ONLY_MODE
            OCR_ONLY_MODE = True
            return False

    def extract_text_with_ocr(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as string
        """
        try:
            # Perform OCR on the image
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            logger.info(f"OCR completed for {image_path} - Extracted {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {str(e)}")
            return ""


class FrameProcessor:
    """
    Class to handle OCR and LLM processing for frames.
    """
    
    def __init__(self, use_key_rotation=False):
        """
        Initialize the frame processor.
        
        Args:
            use_key_rotation: Whether to use API key rotation (currently disabled)
        """
        # Initialize Tesseract OCR
        from pytesseract import pytesseract
        self.pytesseract = pytesseract
        
        # Initialize Gemini processor - prioritize GeminiProcessor over direct API
        global gemini_processor
        self.gemini_processor = gemini_processor
        
        # Initialize direct API model as fallback
        if not self.gemini_processor:
            try:
                self.model = genai.GenerativeModel(GEMINI_PREFERRED_MODEL)
                logger.info(f"Initialized frame processor with Tesseract OCR and model {GEMINI_PREFERRED_MODEL}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {str(e)}")
                logger.warning("⚠️ Will continue attempting to use LLM processing")
        else:
            logger.info(f"Initialized frame processor with Tesseract OCR and GeminiProcessor")
    
    async def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image using OCR with enhanced configuration options.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"OCR ERROR: Image file not found: {image_path}")
                return ""
                
            # Open the image file
            try:
                img = Image.open(image_path)
                logger.info(f"OCR image loaded: {image_path} - Size: {img.size}, Mode: {img.mode}")
            except Exception as img_err:
                logger.error(f"OCR ERROR: Failed to open image: {str(img_err)}")
                return ""
            
            # Try multiple OCR configurations to maximize text detection
            ocr_results = []
            ocr_configs = [
                # Default configuration with page segmentation mode 6 (assume a single uniform block of text)
                {'config': '--psm 6', 'description': 'Default-PSM6'},
                
                # Try page segmentation mode 4 (assume a single column of text)
                {'config': '--psm 4', 'description': 'Single-column-PSM4'},
                
                # Try page segmentation mode 3 (fully automatic page segmentation, but no OSD)
                {'config': '--psm 3', 'description': 'Auto-PSM3'},
                
                # Try with different image processing (improve contrast)
                {'config': '--psm 6 -c preserve_interword_spaces=1', 'description': 'Preserve-spaces'}
            ]
            
            # Process with each configuration
            for config in ocr_configs:
                try:
                    logger.debug(f"Trying OCR config: {config['description']}")
                    ocr_text = await asyncio.to_thread(
                        self.pytesseract.image_to_string,
                        img,
                        lang='eng',
                        config=config['config']
                    )
                    
                    ocr_text = ocr_text.strip()
                    if ocr_text:
                        logger.debug(f"OCR config {config['description']} found {len(ocr_text)} chars")
                        ocr_results.append(ocr_text)
                except Exception as config_err:
                    logger.warning(f"OCR ERROR with config {config['description']}: {str(config_err)}")
            
            # Choose the best result (the one with most characters)
            if ocr_results:
                best_result = max(ocr_results, key=len)
                char_count = len(best_result)
                word_count = len(best_result.split())
                logger.info(f"Best OCR result: {char_count} chars, ~{word_count} words")
                
                # Log a preview of the detected text
                preview = best_result[:100] + "..." if len(best_result) > 100 else best_result
                logger.info(f"OCR text preview: {preview}")
                
                return best_result
            else:
                logger.warning(f"No text extracted from image with any OCR configuration: {image_path}")
                return ""
                
        except Exception as e:
            logger.error(f"OCR ERROR: Unhandled exception extracting text from {image_path}: {e}")
            if hasattr(e, "__traceback__"):
                import traceback
                logger.error(f"OCR Traceback: {traceback.format_exc()}")
            return ""
    
    async def process_with_llm(self, image_path: str, ocr_text: str) -> Dict[str, Any]:
        """
        Process an image and its OCR text with Gemini LLM.
        
        Args:
            image_path: Path to the image file
            ocr_text: Text extracted from the image
            
        Returns:
            Dictionary with LLM analysis results
        """
        # OCR-only mode is disabled, always try LLM processing
        logger.info(f"Processing {os.path.basename(image_path)} with LLM")
        
        # Continue with normal LLM processing
        try:
            if not os.path.exists(image_path):
                logger.error(f"ERROR_STAGE_1: Image file not found for LLM processing: {image_path}")
                return {
                    "filtered_text": "ERROR_STAGE_1: Image file not found",
                    "contains_sensitive_info": False,
                    "processing_error": True
                }
            
            # Log OCR text for debugging
            if ocr_text:
                logger.info(f"OCR Text (first 100 chars): {ocr_text[:100]}")
            else:
                logger.warning(f"Empty OCR text for image: {image_path}")
            
            # Load the image
            try:
                img = Image.open(image_path)
                logger.info(f"Image loaded successfully: {image_path} - Size: {img.size}")
            except Exception as img_err:
                logger.error(f"ERROR_STAGE_2: Failed to load image: {str(img_err)}")
                return {
                    "filtered_text": "ERROR_STAGE_2: Failed to load image",
                    "contains_sensitive_info": False,
                    "processing_error": True
                }
            
            # Prepare the prompt for Gemini with clearer instructions and examples
            prompt = f"""
            Analyze this screen capture with the OCR text.
            
            OCR Text: {ocr_text}
            
            Your task:
            1. Extract all meaningful text visible in the image
            2. You MUST try to identify words and labels even if the OCR text is incomplete or garbled
            3. If there are menus, buttons, or UI elements with text, include them
            4. Remove OCR errors and random artifacts
            5. Flag any sensitive information (passwords, API keys, credentials, personal data)
            6. IMPORTANT: NEVER return "No readable text" unless the image truly has NO text at all
            7. Even partial or single words are better than returning "No readable text"
            8. Organize text with category labels like "UI:", "Content:", "Menu:" etc. followed by text
            9. Use pipe symbols (|) to separate different UI elements and sections
            
            RESPONSE FORMAT:
            You must respond EXCLUSIVELY with a valid JSON object that follows this exact schema:
            {{
                "filtered_text": string,  // ALL text found in the image, formatted with sections and separators
                "contains_sensitive_info": boolean,  // Must be exactly true or false
                "sensitive_content_types": string[]  // Array of strings, empty if no sensitive info
            }}
            
            EXAMPLES:
            
            Example 1 (UI elements):
            {{
                "filtered_text": "App: Firefox | Time: 3:45 PM | UI: Home | Dashboard | Settings | Profile | Logout | Section: Search... | Content: Recent Activity: No new notifications",
                "contains_sensitive_info": false,
                "sensitive_content_types": []
            }}
            
            Example 2 (With sensitive info):
            {{
                "filtered_text": "Page: Settings | Section: API Configuration | Key: sk_test_EXAMPLE_KEY_PLACEHOLDER | User: admin@example.com",
                "contains_sensitive_info": true,
                "sensitive_content_types": ["api_key", "email"]
            }}
            
            Example 3 (Even with partial text):
            {{
                "filtered_text": "App: Dashboard | Status: Online | Config: System Options | Menu: File | Edit | View",
                "contains_sensitive_info": false,
                "sensitive_content_types": []
            }}
            
            DO NOT include any explanations, markdown formatting, or code blocks - JUST THE JSON OBJECT.
            """
            
            # Process with GeminiProcessor if available
            if self.gemini_processor:
                try:
                    logger.info(f"Processing with GeminiProcessor for image: {os.path.basename(image_path)}")
                    result = await asyncio.to_thread(
                        self.gemini_processor.process_image_with_text,
                        image_path=image_path,
                        prompt=prompt
                    )
                    
                    if result.success:
                        logger.info(f"GeminiProcessor success with model: {result.model_used}")
                        try:
                            # Parse the JSON response
                            result_json = json.loads(result.content)
                            logger.info(f"Successfully parsed JSON response from GeminiProcessor")
                            return result_json
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Failed to parse GeminiProcessor JSON response: {str(json_err)}")
                            # Continue to direct API as fallback
                    else:
                        logger.error(f"GeminiProcessor failed: {result.error}")
                        # Continue to direct API as fallback
                except Exception as processor_err:
                    logger.error(f"GeminiProcessor error: {str(processor_err)}")
                    # Continue to direct API as fallback
            
            # If GeminiProcessor failed or isn't available, use direct API
            logger.info(f"Calling Gemini API directly for image: {os.path.basename(image_path)}")
            try:
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [prompt, img]
                )
                logger.info(f"Gemini API call successful, response length: {len(response.text) if response and hasattr(response, 'text') else 'unknown'}")
            except Exception as api_err:
                logger.error(f"ERROR_STAGE_3: Gemini API call failed: {str(api_err)}")
                # Try fallback models before giving up
                fallback_success = False
                for fallback_model in GEMINI_FALLBACK_MODELS:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        fallback = genai.GenerativeModel(fallback_model)
                        response = await asyncio.to_thread(
                            fallback.generate_content,
                            [prompt, img]
                        )
                        logger.info(f"Fallback to {fallback_model} succeeded")
                        fallback_success = True
                        break
                    except Exception as fallback_err:
                        logger.error(f"Fallback to {fallback_model} failed: {str(fallback_err)}")
                        continue
                
                if not fallback_success:
                    # Fall back to OCR-only mode if all API calls fail
                    logger.warning(f"⚠️ Falling back to OCR-only mode for this frame due to API error")
                    return {
                        "filtered_text": ocr_text if ocr_text else f"ERROR_STAGE_3: Gemini API call failed and no OCR text available",
                        "contains_sensitive_info": False,
                        "processing_error": True,
                        "ocr_only_mode": True
                    }
            
            # Parse the response
            try:
                text = response.text
                logger.debug(f"Raw response (truncated): {text[:200]}...")
                
                # Extract JSON part
                if "```json" in text:
                    json_str = text.split("```json")[1].split("```")[0].strip()
                    logger.debug("JSON extracted from markdown code block with json language specified")
                elif "```" in text:
                    json_str = text.split("```")[1].strip()
                    logger.debug("JSON extracted from generic markdown code block")
                else:
                    json_str = text.strip()
                    logger.debug("Using raw response as JSON")
                
                result = json.loads(json_str)
                logger.info(f"Successfully parsed JSON response")
            except Exception as parse_err:
                logger.error(f"ERROR_STAGE_4: Failed to parse Gemini response: {str(parse_err)}")
                logger.error(f"Raw response was: {text[:500]}...")
                # Fall back to OCR-only mode
                logger.warning(f"⚠️ Falling back to OCR-only mode for this frame due to parsing error")
                return {
                    "filtered_text": ocr_text if ocr_text else f"ERROR_STAGE_4: Failed to parse response and no OCR text available",
                    "contains_sensitive_info": False,
                    "processing_error": True,
                    "ocr_only_mode": True
                }
            
            # Validate result content
            if not result.get("filtered_text") or result.get("filtered_text").strip() == "":
                logger.warning(f"Model returned empty filtered_text for {image_path}")
                result["filtered_text"] = "No readable text (Empty model response)"
            elif result.get("filtered_text").strip() == "No readable text" and ocr_text and len(ocr_text.strip()) > 20:
                logger.warning(f"Model returned 'No readable text' but OCR found {len(ocr_text.strip())} chars of text")
                # Include a sample of the OCR text in the result for debugging
                result["filtered_text"] = f"POSSIBLE OCR MISMATCH: Model found no text, but OCR found: {ocr_text[:100]}"
                
            # Log successful processing
            logger.info(f"Completed LLM processing for {image_path} - Found text: {result.get('filtered_text', '')[:50]}...")
            return result
            
        except Exception as e:
            error_message = f"ERROR_STAGE_5: Unhandled error processing {image_path}: {str(e)}"
            logger.error(error_message)
            if hasattr(e, "__traceback__"):
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fall back to OCR text in case of any error
            return {
                "filtered_text": ocr_text if ocr_text else error_message,
                "contains_sensitive_info": False,
                "processing_error": True,
                "ocr_only_mode": True
            }


class FrameProcessorByPath:
    """
    Main class to handle the workflow of processing frames by path.
    """
    
    def __init__(self, batch_size=10):
        """
        Initialize the processor.
        
        Args:
            batch_size: Number of frames to process in each batch
        """
        self.batch_size = batch_size
        self.airtable = AirtableConnector(AIRTABLE_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
        self.frame_processor = FrameProcessor(use_key_rotation=GEMINI_USE_KEY_ROTATION)
        
        # Ensure OCR results directory exists
        os.makedirs(OCR_RESULTS_DIR, exist_ok=True)
        logger.info(f"Initialized FrameProcessorByPath with batch size {batch_size}")
        
        # Report on GeminiProcessor status
        if self.frame_processor.gemini_processor:
            model_info = self.frame_processor.gemini_processor.get_model_info(GEMINI_PREFERRED_MODEL)
            if model_info:
                logger.info(f"Using GeminiProcessor with model: {model_info.name} ({model_info.generation})")
                logger.info(f"Model supports: input tokens: {model_info.input_token_limit}, output tokens: {model_info.output_token_limit}")
            else:
                logger.info(f"Using GeminiProcessor with model: {GEMINI_PREFERRED_MODEL} (details unavailable)")
        elif OCR_ONLY_MODE:
            logger.info("Using OCR-only mode (no LLM processing)")
        else:
            logger.info(f"Using direct API with model: {GEMINI_PREFERRED_MODEL}")
    
    async def process_single_frame(self, folder_path: str) -> Dict[str, Any]:
        """
        Process a single frame based on its folder path.
        
        Args:
            folder_path: Full path to the frame
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"============ PROCESSING FRAME: {folder_path} ============")
            start_time = time.time()
            
            # Find the record in Airtable
            logger.info(f"Step 1: Finding Airtable record for path")
            records = await self.airtable.find_records_by_folder_path(folder_path)
            
            if not records:
                logger.error(f"No Airtable record found for path: {folder_path}")
                return {
                    "status": "error",
                    "error": "No matching Airtable record found",
                    "folder_path": folder_path
                }
            
            record = records[0]  # Take the first matching record
            record_id = record['id']
            logger.info(f"Found Airtable record: {record_id}")
            
            # Extract text with OCR
            logger.info(f"Step 2: Extracting text with OCR")
            ocr_start_time = time.time()
            ocr_text = await self.frame_processor.extract_text(folder_path)
            ocr_time = time.time() - ocr_start_time
            
            ocr_stats = {
                "chars": len(ocr_text) if ocr_text else 0,
                "words": len(ocr_text.split()) if ocr_text else 0,
                "time_seconds": round(ocr_time, 2)
            }
            
            if not ocr_text:
                logger.warning(f"No OCR text extracted from {folder_path}")
            else:
                logger.info(f"OCR completed in {ocr_stats['time_seconds']}s, found {ocr_stats['chars']} chars, {ocr_stats['words']} words")
                
            # Process with LLM
            logger.info(f"Step 3: Processing with LLM")
            llm_start_time = time.time()
            llm_result = await self.frame_processor.process_with_llm(folder_path, ocr_text)
            llm_time = time.time() - llm_start_time
            
            filtered_text = llm_result.get("filtered_text", "No readable text")
            
            # Generate processing stats for diagnostic purposes
            llm_stats = {
                "chars": len(filtered_text) if filtered_text else 0,
                "words": len(filtered_text.split()) if filtered_text else 0,
                "time_seconds": round(llm_time, 2),
                "contains_sensitive_info": llm_result.get("contains_sensitive_info", False)
            }
            
            logger.info(f"LLM completed in {llm_stats['time_seconds']}s, extracted {llm_stats['chars']} chars, {llm_stats['words']} words")
            
            # Prepare OCR data summary with added diagnostic information
            ocr_summary = {
                "processed_at": datetime.datetime.now().isoformat(),
                "status": "processed",
                "ocr_text": llm_result.get("filtered_text", "No readable text"),
                "contains_sensitive_info": llm_result.get("contains_sensitive_info", False),
                "sensitive_content_types": llm_result.get("sensitive_content_types", []),
                "processing_stats": {
                    "ocr": ocr_stats,
                    "llm": llm_stats,
                    "total_time_seconds": round(time.time() - start_time, 2)
                }
            }
            
            # Save OCR result as JSON file
            logger.info(f"Step 4: Saving results to JSON")
            frame_name = os.path.basename(folder_path)
            frame_id = os.path.splitext(frame_name)[0]
            json_path = os.path.join(OCR_RESULTS_DIR, f"{frame_id}.json")
            
            with open(json_path, 'w') as f:
                json.dump({
                    "frame_path": folder_path,
                    "frame_name": frame_name,
                    "airtable_id": record_id,
                    "ocr_data": ocr_summary,
                    "processed_at": datetime.datetime.now().isoformat(),
                    "raw_ocr_text": ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text  # Save original OCR for debugging
                }, f, indent=2)
            
            logger.info(f"Saved OCR result to {json_path}")
            
            # Update Airtable record
            logger.info(f"Step 5: Updating Airtable record")
            sensitive_flag = llm_result.get("contains_sensitive_info", False)
            # Set Flagged field to simple boolean string
            sensitive_flag_value = 'true' if sensitive_flag else 'false'
            
            # Store only the filtered text in OCRData, not the JSON structure
            filtered_text = llm_result.get("filtered_text", "No readable text")
            update_data = {
                "OCRData": filtered_text,
                "Flagged": sensitive_flag_value,
                "ProcessingTime": str(round(time.time() - start_time, 2)) + "s"  # Store processing time in Airtable
            }
            
            # Add detailed sensitivity information to SensitivityConcerns field if sensitive
            if sensitive_flag:
                sensitive_types = llm_result.get("sensitive_content_types", [])
                if sensitive_types:
                    update_data["SensitivityConcerns"] = f"Sensitive content detected: {', '.join(sensitive_types)}"
            
            success = await self.airtable.update_record(record_id, update_data)
            
            if success:
                logger.info(f"Successfully updated Airtable record for {frame_name}")
                
                # Include processing stats in the result
                total_time = time.time() - start_time
                logger.info(f"============ FRAME PROCESSING COMPLETED in {round(total_time, 2)}s ============")
                
                return {
                    "status": "success",
                    "folder_path": folder_path,
                    "record_id": record_id,
                    "sensitive": sensitive_flag,
                    "processing_time": round(total_time, 2),
                    "char_count": llm_stats["chars"],
                    "had_ocr_text": ocr_stats["chars"] > 0
                }
            else:
                logger.error(f"Failed to update Airtable record for {frame_name}")
                return {
                    "status": "error",
                    "error": "Failed to update Airtable record",
                    "folder_path": folder_path,
                    "record_id": record_id
                }
                
        except Exception as e:
            logger.error(f"Error processing frame {folder_path}: {e}")
            if hasattr(e, "__traceback__"):
                import traceback
                logger.error(f"Frame processing traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e),
                "folder_path": folder_path
            }
    
    async def process_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of frames from Airtable records.
        
        Args:
            records: List of Airtable records
            
        Returns:
            Dictionary with batch processing results
        """
        try:
            logger.info(f"Processing batch of {len(records)} records")
            
            results = []
            updates = []
            
            for record in records:
                record_id = record['id']
                fields = record['fields']
                folder_path = fields.get('FolderPath', '')
                
                if not folder_path:
                    logger.warning(f"Record {record_id} has no FolderPath, skipping")
                    results.append({
                        "status": "error",
                        "error": "Missing FolderPath",
                        "record_id": record_id
                    })
                    continue
                
                try:
                    # Extract text with OCR
                    ocr_text = await self.frame_processor.extract_text(folder_path)
                    
                    if not ocr_text:
                        logger.warning(f"No OCR text extracted from {folder_path}")
                    
                    # Process with LLM
                    llm_result = await self.frame_processor.process_with_llm(folder_path, ocr_text)
                    
                    # Prepare OCR data summary
                    ocr_summary = {
                        "processed_at": datetime.datetime.now().isoformat(),
                        "status": "processed",
                        "ocr_text": llm_result.get("filtered_text", "No readable text"),
                        "contains_sensitive_info": llm_result.get("contains_sensitive_info", False),
                        "sensitive_content_types": llm_result.get("sensitive_content_types", [])
                    }
                    
                    # Save OCR result as JSON file
                    frame_name = os.path.basename(folder_path)
                    frame_id = os.path.splitext(frame_name)[0]
                    json_path = os.path.join(OCR_RESULTS_DIR, f"{frame_id}.json")
                    
                    with open(json_path, 'w') as f:
                        json.dump({
                            "frame_path": folder_path,
                            "frame_name": frame_name,
                            "airtable_id": record_id,
                            "ocr_data": ocr_summary,
                            "processed_at": datetime.datetime.now().isoformat()
                        }, f, indent=2)
                    
                    # Prepare Airtable update
                    sensitive_flag = llm_result.get("contains_sensitive_info", False)
                    # Set Flagged field to simple boolean string
                    sensitive_flag_value = 'true' if sensitive_flag else 'false'
                    
                    # Store only the filtered text in OCRData, not the JSON structure
                    filtered_text = llm_result.get("filtered_text", "No readable text")
                    
                    # Prepare update fields
                    update_fields = {
                        "OCRData": filtered_text,
                        "Flagged": sensitive_flag_value
                    }
                    
                    # Add detailed sensitivity information to SensitivityConcerns field if sensitive
                    if sensitive_flag:
                        sensitive_types = llm_result.get("sensitive_content_types", [])
                        if sensitive_types:
                            update_fields["SensitivityConcerns"] = f"Sensitive content detected: {', '.join(sensitive_types)}"
                    
                    updates.append({
                        "id": record_id,
                        "fields": update_fields
                    })
                    
                    results.append({
                        "status": "success",
                        "folder_path": folder_path,
                        "record_id": record_id,
                        "sensitive": sensitive_flag
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing record {record_id}: {e}")
                    results.append({
                        "status": "error",
                        "error": str(e),
                        "record_id": record_id,
                        "folder_path": folder_path
                    })
            
            # Batch update Airtable
            if updates:
                success = await self.airtable.batch_update_records(updates)
                logger.info(f"Batch update of {len(updates)} records: {'Successful' if success else 'Failed'}")
            
            return {
                "status": "completed",
                "total": len(records),
                "successful": sum(1 for r in results if r.get("status") == "success"),
                "errors": sum(1 for r in results if r.get("status") == "error"),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return {
                "status": "error",
                "error": str(e),
                "total": len(records),
                "successful": 0,
                "errors": len(records)
            }
    
    async def process_by_pattern(self, pattern: str, limit: int = None) -> Dict[str, Any]:
        """
        Process frames matching a path pattern.
        
        Args:
            pattern: Path pattern to match
            limit: Maximum number of records to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing frames by pattern: {pattern}")
            
            # Find records in Airtable
            records = await self.airtable.find_records_by_path_pattern(pattern, limit)
            
            if not records:
                logger.warning(f"No Airtable records found for pattern: {pattern}")
                return {
                    "status": "completed",
                    "total": 0,
                    "message": "No matching records found"
                }
            
            logger.info(f"Found {len(records)} records matching pattern")
            
            # Process in batches
            all_results = {
                "status": "completed",
                "total": len(records),
                "batches": 0,
                "successful": 0,
                "errors": 0,
                "batch_results": []
            }
            
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i+self.batch_size]
                batch_result = await self.process_batch(batch)
                all_results["batches"] += 1
                all_results["successful"] += batch_result.get("successful", 0)
                all_results["errors"] += batch_result.get("errors", 0)
                all_results["batch_results"].append(batch_result)
                
                # Save intermediate results
                summary_path = os.path.join(OCR_RESULTS_DIR, f"pattern_processing_{datetime.datetime.now().strftime('%Y%m%d')}.json")
                with open(summary_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
            
            logger.info(f"Completed processing {len(records)} records in {all_results['batches']} batches")
            logger.info(f"Successful: {all_results['successful']}, Errors: {all_results['errors']}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error processing by pattern {pattern}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "pattern": pattern
            }

    async def process_specific_ids(self, record_ids: List[str], path_pattern: str) -> Dict[str, Any]:
        """
        Process frames with specific record IDs.
        
        Args:
            record_ids: List of record IDs to process
            path_pattern: Pattern to match in FolderPath field (for image path)
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing {len(record_ids)} specific record IDs")
            
            # Find records in Airtable
            records = await self.airtable.find_records_by_ids(record_ids)
            
            if not records:
                logger.warning(f"No Airtable records found for the specified IDs")
                return {
                    "status": "completed",
                    "total": 0,
                    "message": "No matching records found"
                }
            
            logger.info(f"Found {len(records)} records for the specific IDs")
            
            # Extract directory from path pattern for use with relative image paths
            base_dir = os.path.dirname(path_pattern) if '*' in path_pattern else path_pattern
            
            # Process in batches
            all_results = {
                "status": "completed",
                "total": len(records),
                "batches": 0,
                "successful": 0,
                "errors": 0,
                "batch_results": []
            }
            
            for i in range(0, len(records), self.batch_size):
                batch = records[i:i+self.batch_size]
                batch_result = await self.process_batch(batch)
                all_results["batches"] += 1
                all_results["successful"] += batch_result.get("successful", 0)
                all_results["errors"] += batch_result.get("errors", 0)
                all_results["batch_results"].append(batch_result)
                
                # Save intermediate results
                summary_path = os.path.join(OCR_RESULTS_DIR, f"specific_ids_processing_{datetime.datetime.now().strftime('%Y%m%d')}.json")
                with open(summary_path, 'w') as f:
                    json.dump(all_results, f, indent=2)
            
            logger.info(f"Completed processing {len(records)} records in {all_results['batches']} batches")
            logger.info(f"Successful: {all_results['successful']}, Errors: {all_results['errors']}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error processing specific IDs: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


async def main():
    """Main entry point for the frame processor."""
    parser = argparse.ArgumentParser(
        description="Process frames by path - OCR and LLM processor for Airtable integration"
    )
    
    # Input options
    parser.add_argument("--folder-path", 
                      help="Path to a specific frame to process")
    parser.add_argument("--folder-path-pattern",
                      help="Pattern to match in FolderPath field (e.g. '/path/to/folder/*.jpg')")
    parser.add_argument("--batch-size", type=int, default=10,
                      help="Number of frames to process in each batch (default: 10)")
    parser.add_argument("--limit", type=int, default=None,
                      help="Maximum number of records to process (default: no limit)")
    parser.add_argument("--specific-ids", 
                      help="Path to a file containing specific record IDs to process (one per line)")
    parser.add_argument("--skip-airtable-update", action="store_true",
                      help="Skip updating Airtable records (results will be saved to JSON files only)")
    parser.add_argument("--model",
                      help="Specify a Gemini model to use (default: from env or gemini-2.0-flash-exp)")
    
    args = parser.parse_args()
    
    if not args.folder_path and not args.folder_path_pattern and not args.specific_ids:
        parser.error("Either --folder-path, --folder-path-pattern, or --specific-ids must be provided")
    
    # Handle model override
    if args.model:
        global GEMINI_PREFERRED_MODEL
        GEMINI_PREFERRED_MODEL = args.model
        logger.info(f"Using command-line specified model: {GEMINI_PREFERRED_MODEL}")
        
        # Reinitialize processor with new model if needed
        global gemini_processor
        if gemini_processor:
            try:
                gemini_processor.default_model = GEMINI_PREFERRED_MODEL
                logger.info(f"Updated GeminiProcessor to use model: {GEMINI_PREFERRED_MODEL}")
            except Exception as model_err:
                logger.error(f"Error updating model: {str(model_err)}")
    
    try:
        # Initialize processor
        processor = FrameProcessorByPath(batch_size=args.batch_size)
        
        if args.specific_ids:
            # Process specific record IDs from file
            logger.info(f"Processing specific record IDs from file: {args.specific_ids}")
            
            # Read record IDs from file
            with open(args.specific_ids, 'r') as f:
                record_ids = [line.strip() for line in f if line.strip()]
            
            if not record_ids:
                logger.error(f"No record IDs found in file: {args.specific_ids}")
                return 1
                
            logger.info(f"Found {len(record_ids)} record IDs to process")
            
            # We still need a path pattern for image paths
            if not args.folder_path_pattern:
                logger.error("When using --specific-ids, you must also provide --folder-path-pattern")
                return 1
                
            result = await processor.process_specific_ids(record_ids, args.folder_path_pattern)
        elif args.folder_path:
            # Process a single frame
            logger.info(f"Processing single frame: {args.folder_path}")
            
            # Modify behavior based on skip-airtable-update flag
            if args.skip_airtable_update:
                # Only run OCR and save to JSON
                logger.info("Skipping Airtable update, only saving OCR results to JSON")
                
                # Extract text with OCR
                ocr_text = await processor.frame_processor.extract_text(args.folder_path)
                
                if not ocr_text:
                    logger.warning(f"No OCR text extracted from {args.folder_path}")
                    
                # Process with LLM
                llm_result = await processor.frame_processor.process_with_llm(args.folder_path, ocr_text)
                
                # Prepare OCR data summary
                ocr_summary = {
                    "processed_at": datetime.datetime.now().isoformat(),
                    "status": "processed",
                    "ocr_text": llm_result.get("filtered_text", "No readable text"),
                    "contains_sensitive_info": llm_result.get("contains_sensitive_info", False),
                    "sensitive_content_types": llm_result.get("sensitive_content_types", [])
                }
                
                # Save OCR result as JSON file
                frame_name = os.path.basename(args.folder_path)
                frame_id = os.path.splitext(frame_name)[0]
                json_path = os.path.join(OCR_RESULTS_DIR, f"{frame_id}.json")
                
                with open(json_path, 'w') as f:
                    json.dump({
                        "frame_path": args.folder_path,
                        "frame_name": frame_name,
                        "ocr_data": ocr_summary,
                        "processed_at": datetime.datetime.now().isoformat()
                    }, f, indent=2)
                
                logger.info(f"Saved OCR result to {json_path}")
                result = {"status": "success", "message": "OCR results saved to JSON", "json_path": json_path}
            else:
                # Regular processing with Airtable update
                result = await processor.process_single_frame(args.folder_path)
        else:
            # Process by pattern
            logger.info(f"Processing frames with pattern: {args.folder_path_pattern}")
            result = await processor.process_by_pattern(args.folder_path_pattern, args.limit)
        
        # Save final results
        final_output = os.path.join(OCR_RESULTS_DIR, f"process_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(final_output, 'w') as f:
            json.dump({
                "processing_time": datetime.datetime.now().isoformat(),
                "parameters": vars(args),
                "results": result
            }, f, indent=2)
            
        logger.info(f"Processing complete. Final results saved to {final_output}")
        return 0
    
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 