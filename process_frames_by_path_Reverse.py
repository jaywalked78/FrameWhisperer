    
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
logger = logging.getLogger("process_frames_by_path_Reverse")

# Load environment variables
load_dotenv()
AIRTABLE_TOKEN = os.environ.get('AIRTABLE_PERSONAL_ACCESS_TOKEN')
AIRTABLE_BASE_ID = os.environ.get('AIRTABLE_BASE_ID')
AIRTABLE_TABLE_NAME = os.environ.get('AIRTABLE_TABLE_NAME', 'tblFrameAnalysis')

# Function to validate a Gemini API key
def validate_gemini_key(api_key):
    """Test if a Gemini API key is valid."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Test")
        return True
    except Exception as e:
        logger.warning(f"API key validation failed: {str(e)}")
        return False

# Get Gemini API key from .env
GEMINI_API_KEY = os.environ.get('GOOGLE_API_KEY_2')

# If not found, use the hardcoded key we found in .env
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "AIzaSyC3SrW9RztB4Vhilkbq_V3i1v9FDWu4z9M"

logger.info(f"Using Gemini API key: {GEMINI_API_KEY[:10]}...")
    
# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
logger.info(f"Configured Gemini with API key")

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
        
        # Initialize Gemini model with the configured API key
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Initialized frame processor with Tesseract OCR and Gemini 1.5 Flash")
    
    async def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return ""
                
            # Open the image file
            img = Image.open(image_path)
            
            # Use Tesseract to extract text
            ocr_text = await asyncio.to_thread(
                self.pytesseract.image_to_string,
                img,
                lang='eng',
                config='--psm 6'
            )
            
            return ocr_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
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
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found for LLM processing: {image_path}")
                return {
                    "filtered_text": "No readable text",
                    "contains_sensitive_info": False,
                    "processing_error": True
                }
            
            # Load the image
            img = Image.open(image_path)
            
            # Prepare the prompt for Gemini with clear schema and examples
            prompt = f"""
            Analyze this screen capture with the OCR text.
            
            OCR Text: {ocr_text}
            
            Your task:
            1. Extract only the meaningful text that was actually on the screen
            2. Ignore garbled text, OCR errors, and random artifacts
            3. Determine if the frame contains sensitive information
            
            RESPONSE FORMAT:
            You must respond EXCLUSIVELY with a valid JSON object that follows this exact schema:
            {{
                "filtered_text": string,  // The cleaned text from the screen or exactly "No readable text" if none found
                "contains_sensitive_info": boolean,  // Must be exactly true or false
                "sensitive_content_types": string[]  // Array of strings, empty if no sensitive info
            }}
            
            EXAMPLES:
            
            Example 1 (No sensitive info):
            {{
                "filtered_text": "Home page - Dashboard - User settings",
                "contains_sensitive_info": false,
                "sensitive_content_types": []
            }}
            
            Example 2 (With sensitive info):
            {{
                "filtered_text": "API Key: sk_test_EXAMPLE_KEY_PLACEHOLDER",
                "contains_sensitive_info": true,
                "sensitive_content_types": ["api_key"]
            }}
            
            Example 3 (No readable text):
            {{
                "filtered_text": "No readable text",
                "contains_sensitive_info": false,
                "sensitive_content_types": []
            }}
            
            DO NOT include any explanations, markdown formatting, or code blocks - JUST THE JSON OBJECT.
            """
            
            # Call Gemini with the image and prompt
            response = await asyncio.to_thread(
                self.model.generate_content,
                [prompt, img]
            )
            
            # Parse the response
            text = response.text
            # Extract JSON part
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].strip()
            else:
                json_str = text.strip()
            
            result = json.loads(json_str)
            
            # Ensure "No readable text" is used when appropriate
            if not result.get("filtered_text") or result.get("filtered_text").strip() == "":
                result["filtered_text"] = "No readable text"
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path} with Gemini: {e}")
            return {
                "filtered_text": "No readable text",
                "contains_sensitive_info": False,
                "processing_error": True
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
    
    async def process_single_frame(self, folder_path: str) -> Dict[str, Any]:
        """
        Process a single frame based on its folder path.
        
        Args:
            folder_path: Full path to the frame
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing single frame: {folder_path}")
            
            # Find the record in Airtable
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
            
            logger.info(f"Saved OCR result to {json_path}")
            
            # Update Airtable record
            sensitive_flag = llm_result.get("contains_sensitive_info", False)
            # Set Flagged field to simple boolean string
            sensitive_flag_value = 'true' if sensitive_flag else 'false'
            
            # Store only the filtered text in OCRData, not the JSON structure
            filtered_text = llm_result.get("filtered_text", "No readable text")
            update_data = {
                "OCRData": filtered_text,
                "Flagged": sensitive_flag_value
            }
            
            # Add detailed sensitivity information to SensitivityConcerns field if sensitive
            if sensitive_flag:
                sensitive_types = llm_result.get("sensitive_content_types", [])
                if sensitive_types:
                    update_data["SensitivityConcerns"] = f"Sensitive content detected: {', '.join(sensitive_types)}"
            
            success = await self.airtable.update_record(record_id, update_data)
            
            if success:
                logger.info(f"Successfully updated Airtable record for {frame_name}")
                return {
                    "status": "success",
                    "folder_path": folder_path,
                    "record_id": record_id,
                    "sensitive": sensitive_flag
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
    
    args = parser.parse_args()
    
    if not args.folder_path and not args.folder_path_pattern and not args.specific_ids:
        parser.error("Either --folder-path, --folder-path-pattern, or --specific-ids must be provided")
    
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