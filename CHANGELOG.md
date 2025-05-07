# Changelog

All notable changes to the OCR Pipeline project will be documented in this file.

## [1.1.0] - 2025-05-07

### Added
- Added missing `get_model_info()` method to GeminiProcessor class
- Added API validation caching to prevent redundant model checks between frames
- Added new structured text formatting with categories and separators
- Added `test_ocr_formatting.py` script to demonstrate improved text structure
- Added comprehensive documentation in `OCR_MODE_REMOVED.md`

### Changed
- Modified LLM prompts to produce better structured text output with separators
- Updated both main scripts (`process_frames_by_path.py` and `process_frames_by_path_Reverse.py`) with improved formatting
- Improved error handling for GeminiProcessor initialization
- Replaced random jumbled text with organized categories and pipe separators
- Added global flag to track API validation status for better performance

### Removed
- Permanently removed OCR-Only mode as a processing option
- Removed redundant API validation calls between frame processing

## [1.0.0] - 2025-05-01

### Added
- Initial release of OCR Pipeline
- Core functionality for OCR text extraction and LLM analysis
- Integration with Airtable for data storage
- Sequential and reverse order processing options
- Parallel processing with worker processes
- Webhook notifications
- API key rotation
- Error handling and retry logic 