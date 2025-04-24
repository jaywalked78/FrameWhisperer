#!/usr/bin/env node
/**
 * Robust OCR Worker Process
 * 
 * A more reliable worker script with timeouts, retry logic, and better error handling
 * 
 * Usage:
 *   node robust_ocr_worker.js --frames-file /path/to/frames.txt --worker-id 1 --api-key YOUR_API_KEY
 */

const fs = require('fs');
const path = require('path');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const { execSync, exec } = require('child_process');
const Airtable = require('airtable');
require('dotenv').config();

// Hardcoded paths
const TEMP_DIR = '/tmp/database_tokenizer';
const LOG_DIR = '/home/jason/Documents/DatabaseAdvancedTokenizer/logs';

// Ensure directories exist
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
  console.log(`Created temp directory: ${TEMP_DIR}`);
}

if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  console.log(`Created logs directory: ${LOG_DIR}`);
}

// Parse command line arguments
const argv = yargs(hideBin(process.argv))
  .option('frames-file', {
    alias: 'f',
    description: 'File containing list of frame paths to process',
    type: 'string',
    demandOption: true
  })
  .option('worker-id', {
    alias: 'i',
    description: 'Worker ID (used to isolate work)',
    type: 'number',
    default: 0
  })
  .option('api-key', {
    alias: 'k',
    description: 'Airtable personal access token (not an API key)',
    type: 'string',
    demandOption: true
  })
  .option('base-id', {
    alias: 'b',
    description: 'Airtable base ID',
    type: 'string',
    demandOption: true
  })
  .option('table-name', {
    alias: 't',
    description: 'Airtable table name',
    type: 'string',
    default: 'tblFrameAnalysis'
  })
  .option('timeout', {
    alias: 'o',
    description: 'OCR timeout in seconds',
    type: 'number',
    default: 60
  })
  .option('max-retries', {
    alias: 'r',
    description: 'Maximum retry attempts',
    type: 'number',
    default: 3
  })
  .help()
  .argv;

// Extract arguments
const framesFile = argv['frames-file'];
const workerId = argv['worker-id'];
const apiKey = argv['api-key'];
const baseId = argv['base-id'];
const tableName = argv['table-name'];
const ocrTimeout = (argv['timeout'] || 60) * 1000; // Default to 60 seconds if undefined
const maxRetries = argv['max-retries'];

// Log all arguments for debugging
log(`Arguments:`);
log(`- frames-file: ${framesFile}`);
log(`- worker-id: ${workerId}`);
log(`- base-id: ${baseId}`);
log(`- table-name: ${tableName}`);
log(`- timeout: ${argv['timeout']} seconds (${ocrTimeout} ms)`);
log(`- max-retries: ${maxRetries}`);

// Validate inputs
if (!fs.existsSync(framesFile)) {
  console.error(`Error: Frames file does not exist: ${framesFile}`);
  process.exit(1);
}

// Initialize Airtable client with personal access token instead of API key
const base = new Airtable({
  apiKey: apiKey,
  endpointUrl: 'https://api.airtable.com',
  requestTimeout: 300000  // 5 minute timeout
}).base(baseId);

const table = base(tableName);

log(`Initialized Airtable connection to base ${baseId}, table ${tableName}`);
log(`Using personal access token: ${apiKey ? apiKey.substring(0, 8) + '...' : 'not provided'}`);

// Constants
const OCR_DATA_FIELD = 'OCRData';
const FLAGGED_FIELD = 'Flagged';
const FOLDER_PATH_FIELD = 'FolderPath';
const SENSITIVITY_CONCERNS_FIELD = 'SensitivityConcerns';
const BATCH_SIZE = 10;

// Log function with timestamp and worker ID
function log(message) {
  const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
  console.log(`[${timestamp}] [Worker ${workerId}] ${message}`);
}

/**
 * Read the list of frame paths from the file
 */
function readFramePaths() {
  try {
    const content = fs.readFileSync(framesFile, 'utf8');
    const paths = content.split('\n').filter(line => line.trim());
    log(`Read ${paths.length} frame paths from file`);
    
    // Validate that frames actually exist
    const validPaths = paths.filter(p => {
      const exists = fs.existsSync(p);
      if (!exists) {
        log(`WARNING: Frame file does not exist: ${p}`);
      }
      return exists;
    });
    
    log(`${validPaths.length} of ${paths.length} frames exist on disk`);
    return validPaths;
  } catch (error) {
    log(`Error reading frames file: ${error.message}`);
    return [];
  }
}

/**
 * Verify Airtable configuration and check for duplicates
 */
async function verifyAirtableSetup() {
  // Check table access and schema
  log(`Verifying Airtable connection to base ${baseId}, table ${tableName}`);
  
  try {
    // Check that we can access the table
    const testQuery = await table.select({ maxRecords: 1 }).firstPage();
    log(`Successfully connected to Airtable. Table contains records: ${testQuery.length > 0 ? 'Yes' : 'No'}`);
    
    // Get record counts for each folder to detect potential issues
    log('Checking for potential duplicate records or path issues...');
    
    // Extract unique folder names from the frame paths
    const frames = readFramePaths();
    const folderMap = new Map();
    
    // Group frames by folder
    for (const framePath of frames) {
      const parts = framePath.split('/');
      let folderName = '';
      
      // Find the screen_recording folder part
      for (const part of parts) {
        if (part.startsWith('screen_recording_')) {
          folderName = part;
          break;
        }
      }
      
      if (folderName) {
        if (!folderMap.has(folderName)) {
          folderMap.set(folderName, []);
        }
        folderMap.get(folderName).push(framePath);
      }
    }
    
    // Check each folder in Airtable
    for (const [folderName, folderFrames] of folderMap.entries()) {
      try {
        // Count records for this folder
        const folderQuery = `FIND("${folderName}", {${FOLDER_PATH_FIELD}}) > 0`;
        const folderRecords = await table.select({
          filterByFormula: folderQuery,
          fields: [FOLDER_PATH_FIELD],
          maxRecords: 1000
        }).all();
        
        log(`Folder "${folderName}": Found ${folderRecords.length} records in Airtable, ${folderFrames.length} frames in current job`);
        
        // Check for potential duplicates
        if (folderRecords.length > 0) {
          // Create a map of base filenames to record IDs
          const filenameMap = new Map();
          
          for (const record of folderRecords) {
            const recordPath = record.fields[FOLDER_PATH_FIELD];
            if (!recordPath) continue;
            
            const filename = path.basename(recordPath);
            if (!filenameMap.has(filename)) {
              filenameMap.set(filename, []);
            }
            filenameMap.get(filename).push(record.id);
          }
          
          // Check for duplicate filenames
          let duplicatesFound = 0;
          for (const [filename, records] of filenameMap.entries()) {
            if (records.length > 1) {
              duplicatesFound++;
              log(`WARNING: Found ${records.length} records for filename "${filename}" in folder "${folderName}"`);
            }
          }
          
          if (duplicatesFound > 0) {
            log(`⚠️ Detected ${duplicatesFound} potential duplicate filenames in folder "${folderName}"`);
          } else {
            log(`✅ No duplicate filenames detected in folder "${folderName}"`);
          }
        }
      } catch (folderError) {
        log(`Error checking folder "${folderName}": ${folderError.message}`);
      }
    }
    
    return true;
  } catch (error) {
    log(`❌ Airtable verification failed: ${error.message}`);
    if (error.error) {
      log(`Error details: ${JSON.stringify(error.error)}`);
    }
    return false;
  }
}

/**
 * Retry function with exponential backoff
 */
async function withRetry(fn, attempt = 1) {
  try {
    return await fn();
  } catch (error) {
    // If we've hit the maximum retry count, throw the error
    if (attempt >= maxRetries) {
      log(`Maximum retry attempts (${maxRetries}) reached. Giving up.`);
      throw error;
    }
    
    // Check for specific errors
    const errorMessage = error.message || '';
    
    // Check for Airtable authentication errors
    if (errorMessage.includes('provide valid api key') || 
        errorMessage.includes('invalid api key') ||
        errorMessage.includes('authentication required') ||
        errorMessage.includes('unauthorized')) {
      log(`CRITICAL: Airtable authentication error: ${errorMessage}`);
      log(`Check that your personal access token is valid and has the correct permissions`);
      throw error; // Don't retry auth errors
    }
    
    // Calculate backoff time (exponential with jitter)
    const baseWaitTime = Math.min(1000 * Math.pow(2, attempt), 30000);
    const jitter = Math.floor(Math.random() * 1000);
    const waitTime = baseWaitTime + jitter;
    
    log(`Attempt ${attempt} failed: ${errorMessage}. Retrying in ${Math.round(waitTime/1000)}s...`);
    
    // Wait and try again
    await new Promise(resolve => setTimeout(resolve, waitTime));
    return withRetry(fn, attempt + 1);
  }
}

/**
 * Find record by FolderPath with improved matching - only returns existing records, never creates new ones
 */
async function findRecordByPath(framePath) {
  return withRetry(async () => {
    // Normalize path for consistent matching - handle both Windows and Unix paths
    let normalizedPath = framePath.replace(/\\/g, '/');
    
    // Remove any trailing slash
    normalizedPath = normalizedPath.replace(/\/$/, '');
    
    // Double-escape backslashes and special characters for Airtable formula
    const escapedPath = normalizedPath.replace(/"/g, '\\"');
    
    log(`Looking for record with FolderPath: "${normalizedPath}"`);
    
    // First try exact match using JIRA-style query (most accurate)
    try {
      // Query Airtable for an exact match
    const records = await table.select({
        filterByFormula: `{${FOLDER_PATH_FIELD}} = "${escapedPath}"`,
        maxRecords: 10
    }).firstPage();
    
    // If found, return the record ID
    if (records.length > 0) {
        log(`Found ${records.length} existing record(s) for: ${normalizedPath}`);
      return records[0].id;
    }
    
      log(`No exact path match found, trying more flexible matching approaches...`);
      
      // Try matching just the filename (without the path)
      const filename = path.basename(normalizedPath);
      if (filename && filename.length > 3) {
        log(`Trying to find record with exact filename: ${filename}`);
        
        // Search for records with exact filename match
        const filenameRecords = await table.select({
          filterByFormula: `FIND("${filename}", {${FOLDER_PATH_FIELD}}) > 0`,
          maxRecords: 20
        }).firstPage();
        
        if (filenameRecords.length > 0) {
          log(`Found ${filenameRecords.length} records containing filename "${filename}"`);
          
          // If only one match, return it
          if (filenameRecords.length === 1) {
            log(`Single match found by filename: ${filenameRecords[0].fields[FOLDER_PATH_FIELD]}`);
            return filenameRecords[0].id;
          }
        }
      }
      
      // If no exact match, try a more flexible approach with the basename
      const basename = path.basename(normalizedPath);
      if (basename && basename.length > 3) {
        log(`Trying to find record with basename: ${basename}`);
        
        // Search for records containing the basename
        const baseRecords = await table.select({
          filterByFormula: `FIND("${basename}", {${FOLDER_PATH_FIELD}}) > 0`,
          maxRecords: 20
        }).firstPage();
        
        if (baseRecords.length > 0) {
          // Double-check to find the closest match
          let bestMatch = null;
          let bestMatchScore = 0;
          
          for (const record of baseRecords) {
            const recordPath = record.fields[FOLDER_PATH_FIELD];
            if (!recordPath) continue;
            
            // Calculate similarity - basic check for now
            let similarity = 0;
            const recordBasename = path.basename(recordPath);
            
            if (recordBasename === basename) similarity += 10;
            if (recordPath.includes(normalizedPath)) similarity += 20;
            if (normalizedPath.includes(recordPath)) similarity += 15;
            
            // Count matching directories
            const recordDirs = recordPath.split('/');
            const frameDirs = normalizedPath.split('/');
            for (let i = 0; i < Math.min(recordDirs.length, frameDirs.length); i++) {
              if (recordDirs[i] === frameDirs[i]) similarity += 2;
            }
            
            // Check if frame number patterns match (e.g., frame_000123.jpg)
            const frameNumberPattern = /frame_(\d+)\.jpg$/i;
            const frameMatch = normalizedPath.match(frameNumberPattern);
            const recordMatch = recordPath.match(frameNumberPattern);
            
            if (frameMatch && recordMatch && frameMatch[1] === recordMatch[1]) {
              similarity += 15; // Strong boost for matching frame numbers
            }
            
            // Look for recording date pattern matches
            const datePattern = /recording_(\d{4}_\d{2}_\d{2})/i;
            const dateMatch = normalizedPath.match(datePattern);
            const recordDateMatch = recordPath.match(datePattern);
            
            if (dateMatch && recordDateMatch && dateMatch[1] === recordDateMatch[1]) {
              similarity += 10; // Boost for matching recording date
            }
            
            log(`Record ${record.id} similarity score: ${similarity} for path: ${recordPath}`);
            
            if (similarity > bestMatchScore) {
              bestMatchScore = similarity;
              bestMatch = record;
            }
          }
          
          // Lower the threshold from 15 to 10 for better matches
          if (bestMatch && bestMatchScore > 10) {
            log(`Found similar record with score ${bestMatchScore}: ${bestMatch.fields[FOLDER_PATH_FIELD]}`);
            return bestMatch.id;
          } else if (bestMatch) {
            log(`Found potential match but score (${bestMatchScore}) too low: ${bestMatch.fields[FOLDER_PATH_FIELD]}`);
          }
        }
      }
      
      // Try just the directory name approach as last resort
      const dirName = path.dirname(normalizedPath).split('/').pop();
      if (dirName && dirName.length > 5 && dirName.includes('recording')) {
        log(`Last attempt: Trying to find record with directory name: ${dirName}`);
        
        const dirRecords = await table.select({
          filterByFormula: `FIND("${dirName}", {${FOLDER_PATH_FIELD}}) > 0`,
          maxRecords: 10
        }).firstPage();
        
        if (dirRecords.length > 0) {
          log(`Found ${dirRecords.length} records with directory name ${dirName}`);
          return dirRecords[0].id; // Return the first match as last resort
        }
      }
      
      // If no match found, log and return null - do NOT create a new record
      log(`No matching record found for: ${normalizedPath}. SKIPPING.`);
      return null;
    } catch (error) {
      log(`Error searching for record for ${normalizedPath}: ${error.message}`);
      if (error.error) {
        log(`Detailed error: ${JSON.stringify(error.error)}`);
      }
      return null; // Return null on error to indicate no record found
    }
  });
}

/**
 * Run OCR on a frame using Tesseract with timeout
 */
function runOcr(framePath) {
  return new Promise((resolve, reject) => {
    // Make sure timeout is a valid number (default to 60 seconds)
    const timeoutMs = isNaN(ocrTimeout) ? 60000 : ocrTimeout;
    
    const cmd = `tesseract "${framePath}" stdout`;
    let timeoutId;
    
    log(`Running OCR on frame: ${path.basename(framePath)} with timeout ${timeoutMs/1000} seconds`);
    
    // Execute OCR command
    const process = exec(cmd, {
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 10, // 10MB buffer
    }, (error, stdout, stderr) => {
      clearTimeout(timeoutId);
      
      if (error) {
        if (error.killed) {
          reject(new Error(`OCR process timed out after ${timeoutMs/1000} seconds`));
        } else {
          reject(new Error(`OCR process failed: ${error.message}`));
        }
        return;
      }
      
      const result = stdout.trim();
      log(`OCR complete for ${path.basename(framePath)}: ${result.length} characters extracted`);
      
      // Return the OCR text
      resolve(result);
    });
    
    // Set timeout to kill process if it takes too long
    timeoutId = setTimeout(() => {
      log(`OCR process timed out for ${path.basename(framePath)} after ${timeoutMs/1000} seconds`);
      process.kill();
    }, timeoutMs);
  }).catch(error => {
    log(`OCR error for ${path.basename(framePath)}: ${error.message}`);
    // Return an explicit error message that can be displayed
    return `OCR processing error: ${error.message}`;
  });
}

/**
 * Check if OCR data contains sensitive content
 */
function checkSensitiveContent(ocrData) {
  if (!ocrData) return false;
  
  // List of sensitive keywords to check for
  const sensitiveKeywords = [
    // Personal Information
    'ssn', 'social security', 'passport', 'driver license', 'birth date', 'birthdate', 'date of birth',
    'credit card', 'visa', 'mastercard', 'amex', 'american express', 'discover', 'cvv', 'expiration date',
    'phone number', 'address', 'email', 'password', 'secret', 'confidential', 'private',
    // Financial
    'bank account', 'routing number', 'swift code', 'iban', 'account number', 'pin', 'tax id',
    'transaction', 'balance', 'transfer', 'payment', 'invoice', 'statement',
    // Medical
    'patient id', 'medical record', 'diagnosis', 'treatment', 'prescription', 'health', 'doctor',
    'medication', 'condition', 'symptom', 'illness', 'insurance id'
  ];
  
  // Convert to lowercase for case-insensitive matching
  const lowerOcrData = ocrData.toLowerCase();
  
  // Check for patterns (credit card, SSN, etc.)
  const patterns = [
    { pattern: /\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b/, name: 'ssn pattern' },  // SSN
    { pattern: /\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b/, name: 'credit card pattern' }, // Credit card
    { pattern: /\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b/, name: 'phone pattern' }, // US Phone
    { pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/, name: 'email pattern' } // Email
  ];
  
  // Check for pattern matches
  for (const { pattern, name } of patterns) {
    if (pattern.test(ocrData)) {
      log(`Detected sensitive pattern: ${name}`);
      return true; // Explicit boolean true
    }
  }
  
  // Check for keyword matches
  for (const keyword of sensitiveKeywords) {
    if (lowerOcrData.includes(keyword)) {
      log(`Detected sensitive keyword: ${keyword}`);
      return true; // Explicit boolean true
    }
  }
  
  return false; // Explicit boolean false
}

/**
 * Process OCR text through LLM for structured analysis
 */
async function processWithLLM(ocrText) {
  // Add timeout for LLM processing (60 seconds)
  const LLM_TIMEOUT_MS = 60000;
  let timeoutId;
  
  try {
    const result = await Promise.race([
      new Promise(async (resolve) => {
  try {
    // Import the LLM processing module
    const { processContentWithGemini } = require('./llm_processor');
    
    log('Processing OCR text with LLM...');
    const llmResult = await processContentWithGemini(ocrText);
          
          // Check if we got an empty response
          if (!llmResult || !llmResult.processedContent || llmResult.processedContent.trim() === '') {
            log('WARNING: Received empty response from LLM, using fallback processing');
            const fallbackResult = createFallbackProcessing(ocrText);
            log(`Fallback processing complete - using direct OCR extraction`);
            resolve(fallbackResult);
            return;
          }
    
    log(`LLM processing complete - sensitive: ${llmResult.containsSensitiveInfo}`);
    if (llmResult.sensitivityConcerns && llmResult.sensitivityConcerns.length > 0) {
      log(`Sensitivity concerns: ${llmResult.sensitivityConcerns.join(', ')}`);
    }
    
    // Return only the essential data for Airtable update
          resolve({
      processedContent: llmResult.processedContent || ocrText,
      isSensitive: llmResult.containsSensitiveInfo || false,
      sensitivityConcerns: llmResult.sensitivityConcerns || []
          });
        } catch (innerError) {
          log(`Inner LLM processing error: ${innerError.message}`);
          log('Using fallback processing due to LLM error');
          const fallbackResult = createFallbackProcessing(ocrText);
          resolve(fallbackResult);
        }
      }),
      new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          reject(new Error(`LLM processing timed out after ${LLM_TIMEOUT_MS/1000} seconds`));
        }, LLM_TIMEOUT_MS);
      })
    ]);
    
    return result;
  } catch (error) {
    log(`LLM processing error: ${error.message}`);
    log('Using fallback processing due to timeout');
    return createFallbackProcessing(ocrText);
  } finally {
    if (timeoutId) clearTimeout(timeoutId);
  }
}

/**
 * Create a fallback processing result when LLM fails
 * This extracts basic information directly from the OCR text
 */
function createFallbackProcessing(ocrText) {
  log('Creating fallback processing from OCR text');
  
  // Create a simple object with basic extraction
  let isSensitive = false;
  let sensitivityConcerns = [];
  
  // Check for potentially sensitive content with simple regex patterns
  const sensitivePatterns = [
    { regex: /password/i, concern: 'Password mentioned' },
    { regex: /credit card/i, concern: 'Credit card information' },
    { regex: /\b(?:\d[ -]*?){13,16}\b/, concern: 'Possible credit card number' },
    { regex: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/, concern: 'Email address' },
    { regex: /SSN|social security/i, concern: 'SSN mentioned' },
    { regex: /\b\d{3}[-]?\d{2}[-]?\d{4}\b/, concern: 'Possible SSN format' },
    { regex: /api[_\s-]?key/i, concern: 'API key mentioned' },
    { regex: /secret/i, concern: 'Secret mentioned' },
    { regex: /private/i, concern: 'Private information mentioned' }
  ];
  
  // Check for sensitive patterns
  for (const pattern of sensitivePatterns) {
    if (pattern.regex.test(ocrText)) {
      isSensitive = true;
      sensitivityConcerns.push(pattern.concern);
    }
  }
  
  // Create a simple processed content with clean OCR text
  const processedContent = ocrText.replace(/[\r\n]+/g, ' ').trim();
  
  return {
    processedContent,
    isSensitive,
    sensitivityConcerns: sensitivityConcerns.length > 0 ? sensitivityConcerns : ['Processed with fallback (LLM unavailable)']
  };
}

/**
 * Update Airtable record with processed data - improved handling
 */
async function updateAirtableRecord(recordId, ocrData, processedData) {
  return withRetry(async () => {
    try {
      // Get the current record to see available fields
      const record = await table.find(recordId);
      if (!record) {
        throw new Error(`Record not found: ${recordId}`);
      }
      
      // Get the available field names in this record
      const availableFields = Object.keys(record.fields);
      log(`Available fields in record: ${availableFields.join(', ')}`);
      
      // Check the environment for defined field names
      const fieldConfig = {
        ocrData: process.env.OCR_DATA_FIELD || OCR_DATA_FIELD || 'OCRData',
        flagged: process.env.FLAGGED_FIELD || FLAGGED_FIELD || 'Flagged',
        sensitivityConcerns: process.env.SENSITIVITY_CONCERNS_FIELD || SENSITIVITY_CONCERNS_FIELD || 'SensitivityConcerns',
        summary: process.env.SUMMARY_FIELD || 'Summary',
        toolsVisible: process.env.TOOLS_VISIBLE_FIELD,
        actionsDetected: process.env.ACTIONS_DETECTED_FIELD,
        technicalDetails: process.env.TECHNICAL_DETAILS_FIELD
      };
      
      log(`Field configuration:
- OCR data field: ${fieldConfig.ocrData}
- Flagged field: ${fieldConfig.flagged}
- Sensitivity concerns field: ${fieldConfig.sensitivityConcerns}
- Summary field: ${fieldConfig.summary}
- Tools visible field: ${fieldConfig.toolsVisible}
- Actions detected field: ${fieldConfig.actionsDetected}
- Technical details field: ${fieldConfig.technicalDetails}`);
      
      // Build a safer update payload - only include fields that exist in Airtable
      const updateFields = {};
      
      // Process each field type, with safety checks
      
      // 1. Always include OCR data as a baseline
      if (fieldConfig.ocrData) {
        if (!availableFields.includes(fieldConfig.ocrData)) {
          log(`WARNING: Field '${fieldConfig.ocrData}' not found in Airtable schema. Data may not be saved properly.`);
          log(`Available fields are: ${availableFields.join(', ')}`);
          
          // Try to find an alternative OCR field
          const potentialOcrFields = availableFields.filter(field => 
            field.toLowerCase().includes('ocr') || 
            field.toLowerCase().includes('text') ||
            field.toLowerCase().includes('data')
          );
          
          if (potentialOcrFields.length > 0) {
            log(`Found potential alternative OCR fields: ${potentialOcrFields.join(', ')}`);
            fieldConfig.ocrData = potentialOcrFields[0];
            log(`Using '${fieldConfig.ocrData}' as alternative OCR field`);
          }
        }
        
        // Even if field doesn't exist, we'll try to create it
        updateFields[fieldConfig.ocrData] = 
          typeof processedData.processedContent === 'string' ? 
            processedData.processedContent : 
            String(ocrData || '');
        log(`Adding ${fieldConfig.ocrData} field (${updateFields[fieldConfig.ocrData].length} chars)`);
      }
      
      // 2. Flagged/sensitivity status - coerce to string true/false for compatibility
      if (fieldConfig.flagged) {
        if (availableFields.includes(fieldConfig.flagged)) {
          const isFlagged = Boolean(processedData.isSensitive || processedData.sensitive || false);
          updateFields[fieldConfig.flagged] = isFlagged ? "true" : "false";
          log(`Adding ${fieldConfig.flagged} field: ${updateFields[fieldConfig.flagged]}`);
        } else {
          log(`Skipping field '${fieldConfig.flagged}' - not found in Airtable schema`);
        }
      }
      
      // 3. Sensitivity concerns - join array or use string directly
      if (fieldConfig.sensitivityConcerns) {
        if (availableFields.includes(fieldConfig.sensitivityConcerns)) {
          // Handle both array and string formats
          let concerns = '';
          if (Array.isArray(processedData.sensitivityConcerns)) {
            concerns = processedData.sensitivityConcerns.join(', ');
          } else if (typeof processedData.sensitivityConcerns === 'string') {
            concerns = processedData.sensitivityConcerns;
          } else if (typeof processedData.sensitive === 'boolean' && processedData.sensitive) {
            concerns = 'Marked as sensitive by automated analysis';
          } else {
            concerns = 'Processed with automated analysis';
          }
          updateFields[fieldConfig.sensitivityConcerns] = concerns;
          log(`Adding ${fieldConfig.sensitivityConcerns} field (${concerns.length} chars)`);
        } else {
          log(`Skipping field '${fieldConfig.sensitivityConcerns}' - not found in Airtable schema`);
        }
      }
      
      // 4. Summary - if available and field is configured
      if (fieldConfig.summary && (processedData.summary || processedData.toolsVisible)) {
        if (availableFields.includes(fieldConfig.summary)) {
          // Use the summary directly or build one from toolsVisible if needed
          const summary = processedData.summary || 
            `Shows ${Array.isArray(processedData.toolsVisible) ? 
              processedData.toolsVisible.join(', ') : 
              'database interface'}`;
              
          updateFields[fieldConfig.summary] = summary;
          log(`Adding ${fieldConfig.summary} field (${summary.length} chars)`);
        } else {
          log(`Skipping field '${fieldConfig.summary}' - not found in Airtable schema`);
        }
      }
      
      // 5. Tools visible - if field configured and data available
      if (fieldConfig.toolsVisible && processedData.toolsVisible) {
        if (availableFields.includes(fieldConfig.toolsVisible)) {
          if (Array.isArray(processedData.toolsVisible)) {
            updateFields[fieldConfig.toolsVisible] = processedData.toolsVisible.join(', ');
          } else if (typeof processedData.toolsVisible === 'string') {
            updateFields[fieldConfig.toolsVisible] = processedData.toolsVisible;
          }
          
          if (updateFields[fieldConfig.toolsVisible]) {
            log(`Adding ${fieldConfig.toolsVisible} field (${updateFields[fieldConfig.toolsVisible].length} chars)`);
          }
      } else {
          log(`Skipping field '${fieldConfig.toolsVisible}' - not found in Airtable schema`);
        }
      }
      
      // 6. Actions detected - if field configured and data available
      if (fieldConfig.actionsDetected && processedData.actionsDetected) {
        if (availableFields.includes(fieldConfig.actionsDetected)) {
          if (Array.isArray(processedData.actionsDetected)) {
            updateFields[fieldConfig.actionsDetected] = processedData.actionsDetected.join(', ');
          } else if (typeof processedData.actionsDetected === 'string') {
            updateFields[fieldConfig.actionsDetected] = processedData.actionsDetected;
          }
          
          if (updateFields[fieldConfig.actionsDetected]) {
            log(`Adding ${fieldConfig.actionsDetected} field (${updateFields[fieldConfig.actionsDetected].length} chars)`);
          }
        } else {
          log(`Skipping field '${fieldConfig.actionsDetected}' - not found in Airtable schema`);
        }
      }
      
      // 7. Technical details - if field configured and data available
      if (fieldConfig.technicalDetails && processedData.technicalDetails) {
        if (availableFields.includes(fieldConfig.technicalDetails)) {
          updateFields[fieldConfig.technicalDetails] = String(processedData.technicalDetails);
          log(`Adding ${fieldConfig.technicalDetails} field (${updateFields[fieldConfig.technicalDetails].length} chars)`);
        } else {
          log(`Skipping field '${fieldConfig.technicalDetails}' - not found in Airtable schema`);
        }
      }
      
      // If no fields to update, log warning and return
      if (Object.keys(updateFields).length === 0) {
        log(`Warning: No fields to update for record ${recordId}`);
        return record;
      }
      
      // Log the update summary
      log(`Updating record ${recordId} with ${Object.keys(updateFields).length} fields`);
      Object.keys(updateFields).forEach(field => {
        const value = updateFields[field];
        const displayValue = typeof value === 'string' ? 
          `${value.length} chars` : 
          JSON.stringify(value).substring(0, 30);
        log(`- ${field}: ${displayValue}`);
      });
      
      // Perform the update with error handling
      try {
      const updated = await table.update(recordId, updateFields);
      log(`Record ${recordId} updated successfully`);
      return updated;
      } catch (updateError) {
        // Handle specific Airtable errors
        log(`Error updating record ${recordId}: ${updateError.message}`);
        
        if (updateError.message.includes('Unknown field name')) {
          // Try with just the OCRData field as a fallback
          const fieldName = updateError.message.match(/Unknown field name: "([^"]+)"/);
          if (fieldName && fieldName[1]) {
            log(`Field "${fieldName[1]}" doesn't exist in Airtable schema - will remove it`);
            delete updateFields[fieldName[1]];
            
            // Try again with the reduced fields
            if (Object.keys(updateFields).length > 0) {
              log(`Retrying update with ${Object.keys(updateFields).length} fields`);
              const retryUpdate = await table.update(recordId, updateFields);
              log(`Update succeeded after removing problematic field`);
              return retryUpdate;
            } else {
              log(`No valid fields remain after removing problematic field`);
              return record;
            }
          }
        }
        
        // If all else fails, try with just OCRData or a simpler OCR field
        const ocrFieldName = fieldConfig.ocrData || 'OCRData';
        const ocrContent = typeof processedData.processedContent === 'string' ? 
          processedData.processedContent : 
          String(ocrData || '');
        
        try {
          log(`Attempting minimal update with just '${ocrFieldName}' field`);
          const minimalUpdate = await table.update(recordId, {
            [ocrFieldName]: ocrContent
          });
          log(`Minimal update succeeded with just '${ocrFieldName}' field`);
          return minimalUpdate;
        } catch (minimalError) {
          log(`Even minimal update failed: ${minimalError.message}`);
          
          // Try finding any text field as a last resort
          const textFields = availableFields.filter(field => 
            field.toLowerCase().includes('text') || 
            field.toLowerCase().includes('data') ||
            field.toLowerCase().includes('content')
          );
          
          if (textFields.length > 0) {
            const lastResortField = textFields[0];
            log(`Last resort: Trying to update field '${lastResortField}'`);
            
            try {
              const lastResortUpdate = await table.update(recordId, {
                [lastResortField]: ocrContent
              });
              log(`Last resort update succeeded with field '${lastResortField}'`);
              return lastResortUpdate;
            } catch (lastError) {
              log(`All update attempts failed. Last error: ${lastError.message}`);
              throw lastError;
            }
          } else {
            throw minimalError;
          }
        }
      }
    } catch (error) {
      log(`Failed to update record ${recordId}: ${error.message}`);
      if (error.error) {
        log(`Airtable error details: ${JSON.stringify(error.error)}`);
      }
      throw error;  // Let the retry mechanism handle it
    }
  });
}

/**
 * Process a batch of frames
 */
async function processBatch(frameBatch) {
  log(`Processing batch of ${frameBatch.length} frames`);
  let successCount = 0;
  let skippedCount = 0;
  
  // Add a global batch timeout (10 minutes)
  const batchTimeout = setTimeout(() => {
    log(`WARNING: Batch processing timed out after 10 minutes. This may indicate a hanging operation.`);
    log(`Successfully processed ${successCount} frames, skipped ${skippedCount} frames before timeout.`);
    process.exit(1); // Force exit on timeout to prevent hanging
  }, 10 * 60 * 1000);
  
  try {
    for (let i = 0; i < frameBatch.length; i++) {
      const framePath = frameBatch[i];
      log(`Starting processing for frame ${i+1}/${frameBatch.length}: ${framePath}`);
      const frameStartTime = Date.now();
    
    try {
      // Find the record ID in Airtable by FolderPath
        log(`[${new Date().toISOString()}] Looking up Airtable record for frame: ${framePath}`);
      let recordId = await findRecordByPath(framePath);
        log(`[${new Date().toISOString()}] Airtable lookup completed in ${(Date.now() - frameStartTime)/1000}s`);
        
      if (!recordId) {
          log(`No Airtable record found for path: ${framePath} - SKIPPING`);
          skippedCount++;
          continue; // Skip to the next frame
      }
      
      // Run OCR on the frame with timeout
        log(`[${new Date().toISOString()}] Running OCR for frame: ${framePath}`);
        const ocrStartTime = Date.now();
      const ocrData = await runOcr(framePath);
        log(`[${new Date().toISOString()}] OCR completed in ${(Date.now() - ocrStartTime)/1000}s`);
      
      // Check if OCR returned an error message
      const isOcrError = ocrData.startsWith('OCR processing error:');
      if (isOcrError) {
        log(`OCR returned an error for ${framePath}, will still update record with error message`);
      }
      
      // Process OCR text through LLM
      let processedData;
      try {
          log(`[${new Date().toISOString()}] Starting LLM processing for frame: ${framePath}`);
          const llmStartTime = Date.now();
        processedData = await processWithLLM(ocrData);
          log(`[${new Date().toISOString()}] LLM processing completed in ${(Date.now() - llmStartTime)/1000}s`);
      } catch (llmError) {
        log(`Error during LLM processing: ${llmError.message}`);
        // Create basic processed data with the error
        processedData = {
          processedContent: ocrData,
          isSensitive: false,
          sensitivityConcerns: ['LLM Processing Error']
        };
      }
      
      // Update the record in Airtable with processed data
      try {
          log(`[${new Date().toISOString()}] Updating Airtable record for frame: ${framePath}`);
          const updateStartTime = Date.now();
        await updateAirtableRecord(recordId, ocrData, processedData);
          log(`[${new Date().toISOString()}] Airtable update completed in ${(Date.now() - updateStartTime)/1000}s`);
        // Count as success even if there was an OCR error but we updated the record
        successCount++;
      } catch (updateError) {
        // Handle Airtable update errors more gracefully
        if (updateError.message.includes('Unknown field name')) {
          log(`Airtable schema error: ${updateError.message}`);
          // Try again with only the OCRData field as a fallback
          try {
            log(`Attempting fallback update with just OCRData field`);
            await table.update(recordId, {
                OCRData: String(processedData.processedContent || ocrData)
            });
            log(`Successfully updated record with OCRData only after schema error`);
            successCount++;
          } catch (fallbackError) {
            log(`Even fallback update failed: ${fallbackError.message}`);
          }
        } else {
          log(`Failed to update Airtable: ${updateError.message}`);
        }
      }
        
        log(`Frame ${i+1}/${frameBatch.length} completed in ${(Date.now() - frameStartTime)/1000}s`);
    } catch (error) {
      log(`ERROR processing frame ${framePath}: ${error.message}`);
      if (error.stack) {
        log(`Stack trace: ${error.stack}`);
      }
    }
    
    // Small delay between frames to avoid overwhelming resources
    log(`Completed processing for frame: ${framePath}`);
    await new Promise(resolve => setTimeout(resolve, 100));
    }
  } finally {
    // Always clear the timeout to prevent memory leaks
    clearTimeout(batchTimeout);
  }
  
  log(`Batch complete. Processed ${successCount}/${frameBatch.length} frames successfully. Skipped ${skippedCount} frames with no matching records.`);
  return successCount;
}

/**
 * Process all frames
 */
async function processAllFrames(framePaths) {
  let totalProcessed = 0;
  const totalFrames = framePaths.length;
  
  // Process frames in batches (not using Airtable batch update, but processing frames in groups)
  for (let i = 0; i < totalFrames; i += BATCH_SIZE) {
    const batch = framePaths.slice(i, i + BATCH_SIZE);
    const batchNumber = Math.floor(i/BATCH_SIZE) + 1;
    const totalBatches = Math.ceil(totalFrames/BATCH_SIZE);
    
    log(`Processing batch ${batchNumber} of ${totalBatches}`);
    
    try {
    const processed = await processBatch(batch);
    totalProcessed += processed;
    
    log(`Progress: ${totalProcessed}/${totalFrames} frames (${Math.round(totalProcessed/totalFrames*100)}%)`);
    } catch (batchError) {
      // Log error but continue with next batch instead of failing everything
      log(`ERROR: Batch ${batchNumber} failed with error: ${batchError.message}`);
      log(`Continuing with next batch...`);
      
      if (batchError.stack) {
        log(`Stack trace: ${batchError.stack}`);
      }
    }
    
    // Small delay between batches
    if (i + BATCH_SIZE < totalFrames) {
      log(`Waiting 1 second before next batch...`);
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  return totalProcessed;
}

// Main execution
(async () => {
  log(`Worker ${workerId} starting with OCR timeout: ${ocrTimeout/1000}s, max retries: ${maxRetries}`);
  log(`Reading frames from: ${framesFile}`);
  
  try {
    // Read frame paths
    const framePaths = readFramePaths();
    log(`Found ${framePaths.length} frames to process`);
    
    if (framePaths.length === 0) {
      log('No frames to process. Exiting.');
      process.exit(0);
    }
    
    // Verify Airtable setup before proceeding
    const setupIsValid = await verifyAirtableSetup();
    if (!setupIsValid) {
      log('WARNING: Airtable setup verification detected potential issues.');
      log('Continuing with processing but results may not be as expected.');
    }
    
    // Process all frames
    const startTime = Date.now();
    const totalProcessed = await processAllFrames(framePaths);
    const endTime = Date.now();
    const totalTimeSeconds = (endTime - startTime) / 1000;
    
    log(`Worker ${workerId} completed successfully`);
    log(`Processed ${totalProcessed}/${framePaths.length} frames`);
    log(`Total processing time: ${Math.floor(totalTimeSeconds / 60)}m ${Math.floor(totalTimeSeconds % 60)}s`);
    log(`Average time per frame: ${(totalTimeSeconds / totalProcessed).toFixed(2)}s`);
    
    process.exit(0);
  } catch (error) {
    log(`Fatal error: ${error.message}`);
    process.exit(1);
  }
})();