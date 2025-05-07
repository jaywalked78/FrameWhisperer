#!/usr/bin/env node

/**
 * Script to process all remaining frames in screen recording folders IN REVERSE ORDER
 * 
 * This script will:
 * 1. Sort folders in REVERSE chronological order (Z-A)
 * 2. Exclude folders already marked as finished in Airtable
 * 3. Generate a list of frames from remaining folders
 * 4. Process them using the robust OCR worker with API key rotation
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');
const { promisify } = require('util');
const readFileAsync = promisify(fs.readFile);
const writeFileAsync = promisify(fs.writeFile);
const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);
require('dotenv').config();

// Set variables
const BASE_DIR = '/home/jason/Videos/screenRecordings';
const TEMP_DIR = '/tmp/database_tokenizer';  // Hardcoded absolute path
const LOG_DIR = '/home/jason/Documents/DatabaseAdvancedTokenizer/logs';  // Hardcoded absolute path
const MAX_WORKERS = 3; // Maximum number of concurrent workers

// Create directories if they don't exist
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR, { recursive: true });
  log(`Created temp directory: ${TEMP_DIR}`);
}
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  log(`Created logs directory: ${LOG_DIR}`);
}

// Generate timestamp for this run
const TIMESTAMP = new Date().toISOString().replace(/[:.]/g, '').replace('T', '_').substring(0, 15);
const LOG_FILE = path.join(LOG_DIR, `process_all_frames_reverse_${TIMESTAMP}.log`);

// Log function
function log(message) {
  const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
  const logMessage = `[${timestamp}] ${message}`;
  console.log(logMessage);
  fs.appendFileSync(LOG_FILE, logMessage + '\n');
}

// Validate configuration
function validateConfig() {
  log('Starting to process all remaining frames in REVERSE order');
  log(`Base directory: ${BASE_DIR}`);
  log(`Temp directory: ${TEMP_DIR}`);
  log(`Log directory: ${LOG_DIR}`);
  
  if (!process.env.AIRTABLE_PERSONAL_ACCESS_TOKEN) {
    log('Error: AIRTABLE_PERSONAL_ACCESS_TOKEN not set in .env file');
    process.exit(1);
  }
  
  if (!process.env.AIRTABLE_BASE_ID) {
    log('Error: AIRTABLE_BASE_ID not set in .env file');
    process.exit(1);
  }
  
  // Check for available API keys - prioritize Gemini keys
  const apiKeys = {
    gemini: [],
    google: process.env.GOOGLE_API_KEY
  };
  
  // Check for Gemini API keys in rotational format
  for (let i = 1; i <= 5; i++) {
    if (process.env[`GEMINI_API_KEY_${i}`]) {
      apiKeys.gemini.push(process.env[`GEMINI_API_KEY_${i}`]);
    }
  }
  
  // Also check for a single non-rotational GEMINI_API_KEY
  if (process.env.GEMINI_API_KEY) {
    apiKeys.gemini.unshift(process.env.GEMINI_API_KEY);
  }
  
  // Configure environment based on available keys
  if (apiKeys.gemini.length > 0) {
    log(`Found ${apiKeys.gemini.length} Gemini API keys available for use`);
    
    // Set up the first key as main key if no rotation is needed or available
    if (apiKeys.gemini.length === 1 || MAX_WORKERS === 1) {
      process.env.GEMINI_API_KEY = apiKeys.gemini[0];
      log(`Using single Gemini API key: ${apiKeys.gemini[0].substring(0, 10)}...`);
      process.env.GEMINI_USE_KEY_ROTATION = 'false';
    } else {
      // Set up rotation for multiple workers
      apiKeys.gemini.forEach((key, index) => {
        process.env[`GEMINI_API_KEY_${index+1}`] = key;
      });
      log(`Configured ${apiKeys.gemini.length} Gemini API keys for rotation`);
      process.env.GEMINI_USE_KEY_ROTATION = 'true';
    }
  } 
  // Fall back to Google API key if no Gemini keys are found
  else if (apiKeys.google) {
    log(`No Gemini API keys found, using Google API key as fallback: ${apiKeys.google.substring(0, 10)}...`);
    process.env.GEMINI_API_KEY = apiKeys.google;
    process.env.GEMINI_USE_KEY_ROTATION = 'false';
    
    // If we have multiple workers but only one key, we need to add rate limiting
    if (MAX_WORKERS > 1) {
      process.env.GEMINI_RATE_LIMIT = '20'; // Lower limit for single key with multiple workers
      process.env.GEMINI_COOLDOWN_PERIOD = '90'; // Longer cooldown for single key with multiple workers
      log(`Configured rate limiting for single API key with multiple workers`);
    }
  } else {
    log(`Warning: No API keys found. Processing will attempt to continue with LLM processing.`);
    // OCR_ONLY_MODE is no longer used - we always attempt LLM processing
  }
  
  // Check/set preferred model configuration
  if (process.env.GEMINI_PREFERRED_MODEL) {
    log(`Using preferred Gemini model: ${process.env.GEMINI_PREFERRED_MODEL}`);
  } else {
    // Set default preferred model to Gemini 2.0 Flash experimental
    process.env.GEMINI_PREFERRED_MODEL = 'models/gemini-2.0-flash-exp';
    log(`Setting default preferred model to ${process.env.GEMINI_PREFERRED_MODEL}`);
  }
  
  // Configure fallback models
  process.env.GEMINI_FALLBACK_MODELS = process.env.GEMINI_FALLBACK_MODELS || 'models/gemini-2.0-flash,models/gemini-1.5-flash';
  log(`Configured fallback models: ${process.env.GEMINI_FALLBACK_MODELS}`);
  
  // Always use maximum available workers since we always attempt LLM processing
  const optimalWorkers = Math.min(MAX_WORKERS, Math.max(1, apiKeys.gemini.length || 1));
  log(`Using ${optimalWorkers} worker(s) for processing`);
  
  const trackingTable = process.env.AIRTABLE_TRACKING_TABLE || 'Finished OCR Processed Folders';
  const folderNameField = process.env.FOLDER_NAME_FIELD || 'FolderName';
  const tableName = process.env.AIRTABLE_TABLE_NAME || 'tblFrameAnalysis';
  
  return { trackingTable, folderNameField, tableName, optimalWorkers };
}

// Fetch finished folders from Airtable
async function fetchFinishedFolders(trackingTable, folderNameField) {
  log(`Fetching list of finished folders from Airtable table: ${trackingTable}`);
  
  const AIRTABLE_RESPONSE_FILE = path.join(TEMP_DIR, `airtable_response_reverse_${TIMESTAMP}.json`);
  const FINISHED_FOLDERS_FILE = path.join(TEMP_DIR, `finished_folders_reverse_${TIMESTAMP}.txt`);
  
  try {
    // Ensure the temp directory exists
    if (!fs.existsSync(TEMP_DIR)) {
      fs.mkdirSync(TEMP_DIR, { recursive: true });
      log(`Created temp directory: ${TEMP_DIR}`);
    }
    
    // Execute the curl command and capture the raw response
    const curl = `curl -s -X GET "https://api.airtable.com/v0/${process.env.AIRTABLE_BASE_ID}/${trackingTable}?fields%5B%5D=${folderNameField}" -H "Authorization: Bearer ${process.env.AIRTABLE_PERSONAL_ACCESS_TOKEN}" -H "Content-Type: application/json"`;
    log(`Executing curl command: ${curl.substring(0, 100)}...`);
    const result = execSync(curl);
    
    // Log the response size and save to file
    const responseSize = result.length;
    log(`Received Airtable response: ${responseSize} bytes`);
    fs.writeFileSync(AIRTABLE_RESPONSE_FILE, result);
    log(`Saved Airtable response to: ${AIRTABLE_RESPONSE_FILE}`);
    
    // Parse the response with extra error handling
    let finishedFolders = [];
    try {
      // Check if response is empty
      if (!result || responseSize === 0) {
        log('WARNING: Received empty response from Airtable');
        fs.writeFileSync(FINISHED_FOLDERS_FILE, '');
        return [];
      }
      
      // Print the first 200 characters of the response for debugging
      const responsePreview = result.toString().substring(0, 200).replace(/\n/g, '');
      log(`Response preview: ${responsePreview}...`);
      
      // Try to parse JSON
      const response = JSON.parse(result.toString());
      log(`Successfully parsed JSON response`);
      
      // Extract folder names
      if (response.records && Array.isArray(response.records)) {
        log(`Found ${response.records.length} records in response`);
        
        finishedFolders = response.records
          .filter(record => record.fields && record.fields[folderNameField])
          .map(record => record.fields[folderNameField]);
          
        log(`Extracted ${finishedFolders.length} folder names`);
      } else {
        log(`WARNING: Response does not contain expected 'records' array. Structure: ${JSON.stringify(Object.keys(response))}`);
      }
    } catch (parseError) {
      log(`ERROR parsing Airtable response: ${parseError.message}`);
      log(`Response content (first 500 chars): ${result.toString().substring(0, 500)}`);
      
      // Attempt to manually parse if JSON parsing fails
      log(`Attempting manual extraction with regex...`);
      const folderPattern = new RegExp(`"${folderNameField}":"([^"]*)"`, 'g');
      let match;
      while ((match = folderPattern.exec(result.toString())) !== null) {
        finishedFolders.push(match[1]);
      }
      log(`Manually extracted ${finishedFolders.length} folder names`);
    }
    
    // Write the folder names to a file
    fs.writeFileSync(FINISHED_FOLDERS_FILE, finishedFolders.join('\n'));
    log(`Found ${finishedFolders.length} finished folders in Airtable`);
    log(`Saved folder names to: ${FINISHED_FOLDERS_FILE}`);
    return finishedFolders;
    
  } catch (error) {
    log(`ERROR fetching finished folders: ${error.message}`);
    if (error.stderr) {
      log(`STDERR: ${error.stderr.toString()}`);
    }
    // Create empty finished folders file to avoid further errors
    fs.writeFileSync(FINISHED_FOLDERS_FILE, '');
    return [];
  }
}

// Get all folders sorted in REVERSE chronological order (Z-A)
async function getAllFolders() {
  log('Getting all folders in REVERSE chronological order (Z-A)...');
  
  try {
    // Get all directories in BASE_DIR
    const items = await readdir(BASE_DIR);
    let folders = [];
    
    for (const item of items) {
      const fullPath = path.join(BASE_DIR, item);
      const stats = await stat(fullPath);
      
      if (stats.isDirectory() && item.startsWith('screen_recording_')) {
        folders.push({ path: fullPath, name: item });
      }
    }
    
    // Sort in REVERSE chronological order (Z-A) based on folder name
    folders.sort((a, b) => {
      const partsA = a.name.split('_');
      const partsB = b.name.split('_');
      
      // Compare year, month, day in REVERSE order (Z-A)
      const dateA = partsA[2];
      const dateB = partsB[2];
      if (dateA !== dateB) return dateB.localeCompare(dateA); // Reversed comparison
      
      // Compare hour, minute, second in REVERSE order (Z-A)
      const timeA = `${partsA[4]}_${partsA[5]}_${partsA[6]}`;
      const timeB = `${partsB[4]}_${partsB[5]}_${partsB[6]}`;
      return timeB.localeCompare(timeA); // Reversed comparison
    });
    
    log(`Found ${folders.length} total folders, sorted in reverse order (Z-A)`);
    return folders;
    
  } catch (error) {
    log(`Error getting folders: ${error.message}`);
    return [];
  }
}

// Filter out finished folders
function filterFolders(allFolders, finishedFolders) {
  log('Filtering out finished folders...');
  
  const remainingFolders = allFolders.filter(folder => 
    !finishedFolders.includes(folder.name)
  );
  
  log(`After filtering, ${remainingFolders.length} folders need processing (skipping ${allFolders.length - remainingFolders.length} folders)`);
  return remainingFolders;
}

// Collect frames from folders
async function collectFrames(folders) {
  log('Collecting frames from folders...');
  const REMAINING_FRAMES_FILE = path.join(TEMP_DIR, `remaining_frames_reverse_${TIMESTAMP}.txt`);
  
  let totalFrames = 0;
  let folderCount = 0;
  let framesList = [];
  
  for (const folder of folders) {
    folderCount++;
    log(`Processing folder ${folderCount}/${folders.length}: ${folder.name}`);
    
    try {
      // Find frames recursively in this folder
      const findCmd = `find "${folder.path}" -type f -name "*.jpg"`;
      const frames = execSync(findCmd).toString().trim().split('\n');
      
      // Filter out empty strings
      const validFrames = frames.filter(frame => frame.trim() !== '');
      
      // Sort frames in REVERSE order
      validFrames.sort().reverse();
      
      if (validFrames.length > 0) {
        totalFrames += validFrames.length;
        framesList = framesList.concat(validFrames);
        log(`Added ${validFrames.length} frames from folder: ${folder.name}`);
      } else {
        log(`No frames found in folder: ${folder.name}`);
      }
      
    } catch (error) {
      log(`Error processing folder ${folder.name}: ${error.message}`);
    }
  }
  
  if (totalFrames > 0) {
    log(`Writing ${totalFrames} frames to file: ${REMAINING_FRAMES_FILE}`);
    await writeFileAsync(REMAINING_FRAMES_FILE, framesList.join('\n'));
    log(`Successfully saved frames list`);
  } else {
    log(`No frames found in any remaining folders`);
    await writeFileAsync(REMAINING_FRAMES_FILE, '');
  }
  
  return { framesFile: REMAINING_FRAMES_FILE, totalFrames };
}

// Create batch frame files from master list
async function createBatchFiles(framesFile, totalFrames, numWorkers) {
  log(`Creating ${numWorkers} batch files from master frames list...`);
  
  try {
    // Read the master frames file
    const allFrames = (await readFileAsync(framesFile, 'utf8')).trim().split('\n');
    
    if (allFrames.length === 0 || (allFrames.length === 1 && allFrames[0] === '')) {
      log('No frames to process');
      return [];
    }
    
    // Calculate frames per worker
    const framesPerWorker = Math.ceil(allFrames.length / numWorkers);
    log(`Dividing ${allFrames.length} frames among ${numWorkers} workers (${framesPerWorker} per worker)`);
    
    const batchFiles = [];
    
    // Create a batch file for each worker
    for (let i = 0; i < numWorkers; i++) {
      const start = i * framesPerWorker;
      const end = Math.min(start + framesPerWorker, allFrames.length);
      
      if (start >= allFrames.length) {
        break; // No more frames to process
      }
      
      const workerFrames = allFrames.slice(start, end);
      const batchFile = path.join(TEMP_DIR, `frames_worker_${i+1}_reverse_${TIMESTAMP}.txt`);
      
      await writeFileAsync(batchFile, workerFrames.join('\n'));
      log(`Created batch file for worker ${i+1} with ${workerFrames.length} frames: ${batchFile}`);
      
      batchFiles.push({
        file: batchFile,
        workerId: i+1,
        frameCount: workerFrames.length
      });
    }
    
    return batchFiles;
  } catch (error) {
    log(`Error creating batch files: ${error.message}`);
    return [];
  }
}

// Run worker process to handle one batch of frames
function runWorker(batchFile, workerId, tableName) {
  return new Promise(resolve => {
    log(`Starting worker ${workerId} with batch: ${path.basename(batchFile)}`);
    
    // Build command with all necessary parameters
    const workerCmd = 'node';
    const workerArgs = [
      'robust_ocr_worker.js',
      `--frames-file=${batchFile}`,
      `--worker-id=${workerId}`,
      `--api-key=${process.env.AIRTABLE_PERSONAL_ACCESS_TOKEN}`,
      `--base-id=${process.env.AIRTABLE_BASE_ID}`,
      `--table-name=${tableName}`,
      '--timeout=300',
      '--max-retries=3',
      `--model=${process.env.GEMINI_PREFERRED_MODEL || 'models/gemini-2.0-flash-exp'}`,
      `--fallback-models=${process.env.GEMINI_FALLBACK_MODELS || 'models/gemini-2.0-flash,models/gemini-1.5-flash'}`
    ];
    
    log(`Worker ${workerId} command: ${workerCmd} ${workerArgs.join(' ')}`);
    
    // Clone environment and set needed variables
    const env = Object.assign({}, process.env);
    
    // Remove OCR_ONLY_MODE from environment if set (since we're disabling it)
    if (env.OCR_ONLY_MODE) {
      log(`Note: OCR_ONLY_MODE setting will be ignored for worker ${workerId} as this mode has been disabled.`);
      delete env.OCR_ONLY_MODE;
    }
    
    // Start the worker process
    const worker = spawn(workerCmd, workerArgs, {
      env,
      stdio: 'inherit' // Redirect worker output to this process
    });
    
    log(`Worker ${workerId} started with PID: ${worker.pid}`);
    
    worker.on('close', (code) => {
      log(`Worker ${workerId} exited with code: ${code}`);
      resolve(code);
    });
    
    worker.on('error', (error) => {
      log(`Worker ${workerId} error: ${error.message}`);
      resolve(1); // Error exit code
    });
  });
}

// Run multiple worker processes in parallel
async function runWorkers(batchFiles, tableName) {
  log(`Running ${batchFiles.length} worker processes...`);
  
  // Start all workers in parallel
  const promises = batchFiles.map(batchFile => 
    runWorker(batchFile, batchFile.workerId, tableName)
  );
  
  // Wait for all workers to complete
  const results = await Promise.all(promises);
  
  // Calculate success rate
  const successCount = results.filter(code => code === 0).length;
  log(`Worker completion: ${successCount}/${batchFiles.length} workers completed successfully`);
  
  return successCount === batchFiles.length;
}

// Main execution function
async function main() {
  // Validate configuration first
  const { trackingTable, folderNameField, tableName, optimalWorkers } = validateConfig();
  
  // Use optimal worker count based on available keys
  const numWorkers = optimalWorkers;
  log(`Using ${numWorkers} workers based on available API keys and configuration`);
  
  try {
    // Fetch finished folders from Airtable
    const finishedFolders = await fetchFinishedFolders(trackingTable, folderNameField);
    
    // Get all folders sorted in REVERSE chronological order
    const allFolders = await getAllFolders();
    
    // Filter out finished folders
    const remainingFolders = filterFolders(allFolders, finishedFolders);
    
    if (remainingFolders.length === 0) {
      log('No remaining folders to process. Exiting.');
      return;
    }
    
    // Collect frames from remaining folders
    const { framesFile, totalFrames } = await collectFrames(remainingFolders);
    
    if (totalFrames === 0) {
      log('No frames found in remaining folders. Exiting.');
      return;
    }
    
    // Create batch files for parallel processing
    const batchFiles = await createBatchFiles(framesFile, totalFrames, numWorkers);
    
    if (batchFiles.length === 0) {
      log('No batch files created. Exiting.');
      return;
    }
    
    // Run workers in parallel
    const success = await runWorkers(batchFiles, tableName);
    log(`Processing complete: ${success ? 'All workers succeeded' : 'Some workers failed'}`);
    
  } catch (error) {
    log(`Error in main execution: ${error.message}`);
    if (error.stack) {
      log(`Stack trace: ${error.stack}`);
    }
  }
}

// Start execution
main().catch(error => {
  log(`Unhandled error in main: ${error.message}`);
  process.exit(1);
}); 