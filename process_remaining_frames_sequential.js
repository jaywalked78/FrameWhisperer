#!/usr/bin/env node

/**
 * Script to process all remaining frames in screen recording folders IN REVERSE ORDER, SEQUENTIALLY
 * 
 * This script will:
 * 1. Sort folders in REVERSE chronological order (Z-A)
 * 2. Exclude folders already marked as finished in Airtable
 * 3. Generate a list of frames from remaining folders
 * 4. Process them sequentially (one at a time) for maximum speed
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
const LOG_FILE = path.join(LOG_DIR, `process_all_frames_sequential_${TIMESTAMP}.log`);

// Log function
function log(message) {
  const timestamp = new Date().toISOString().replace('T', ' ').substring(0, 19);
  const logMessage = `[${timestamp}] ${message}`;
  console.log(logMessage);
  fs.appendFileSync(LOG_FILE, logMessage + '\n');
}

// Validate configuration
function validateConfig() {
  log('Starting to process all remaining frames sequentially in REVERSE order');
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
  
  // Check for Gemini API key
  if (!process.env.GEMINI_API_KEY) {
    log('Error: GEMINI_API_KEY not set in .env file. Processing will likely fail.');
    process.exit(1);
  }
  
  const trackingTable = process.env.AIRTABLE_TRACKING_TABLE || 'Finished OCR Processed Folders';
  const folderNameField = process.env.FOLDER_NAME_FIELD || 'FolderName';
  const tableName = process.env.AIRTABLE_TABLE_NAME || 'tblFrameAnalysis';
  
  return { trackingTable, folderNameField, tableName };
}

// Fetch finished folders from Airtable
async function fetchFinishedFolders(trackingTable, folderNameField) {
  log(`Fetching list of finished folders from Airtable table: ${trackingTable}`);
  
  const AIRTABLE_RESPONSE_FILE = path.join(TEMP_DIR, `airtable_response_sequential_${TIMESTAMP}.json`);
  const FINISHED_FOLDERS_FILE = path.join(TEMP_DIR, `finished_folders_sequential_${TIMESTAMP}.txt`);
  
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
  const REMAINING_FRAMES_FILE = path.join(TEMP_DIR, `remaining_frames_sequential_${TIMESTAMP}.txt`);
  
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

// Process a single frame
function processFrame(framePath, frameIndex, totalFrames, tableName) {
  return new Promise((resolve) => {
    log(`Processing frame ${frameIndex+1}/${totalFrames}: ${path.basename(framePath)}`);
    
    // Create a temporary file containing just this one frame
    const tempFrameFile = path.join(TEMP_DIR, `single_frame_${TIMESTAMP}.txt`);
    fs.writeFileSync(tempFrameFile, framePath);
    
    // Build command with all necessary parameters
    const workerCmd = 'node';
    const workerArgs = [
      'robust_ocr_worker.js',
      `--frames-file=${tempFrameFile}`,
      `--worker-id=1`, // Always use worker ID 1 for sequential
      `--api-key=${process.env.AIRTABLE_PERSONAL_ACCESS_TOKEN}`,
      `--base-id=${process.env.AIRTABLE_BASE_ID}`,
      `--table-name=${tableName}`,
      '--timeout=300',
      '--max-retries=3',
      '--sequential=true' // Add flag for sequential processing
    ];
    
    log(`Worker command: ${workerCmd} ${workerArgs.join(' ')}`);
    
    // No environment variables manipulations for API key rotation
    // We just use whatever is in the environment
    
    // Start the process and capture output
    const worker = spawn(workerCmd, workerArgs, {
      stdio: 'pipe' // Capture output for performance analysis
    });
    
    let stdoutData = '';
    let stderrData = '';
    
    worker.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });
    
    worker.stderr.on('data', (data) => {
      stderrData += data.toString();
    });
    
    worker.on('close', (code) => {
      log(`Frame ${frameIndex+1}/${totalFrames} processed with exit code: ${code}`);
      
      // Log completion time
      if (stdoutData.includes("Processing completed in")) {
        const timeMatch = stdoutData.match(/Processing completed in (\d+\.\d+) seconds/);
        if (timeMatch) {
          log(`Frame processing time: ${timeMatch[1]} seconds`);
        }
      }
      
      // Clean up temp file
      try {
        fs.unlinkSync(tempFrameFile);
      } catch (error) {
        log(`Warning: Could not delete temp file: ${error.message}`);
      }
      
      resolve(code === 0);
    });
    
    worker.on('error', (error) => {
      log(`Error processing frame ${frameIndex+1}/${totalFrames}: ${error.message}`);
      resolve(false);
    });
  });
}

// Process all frames sequentially
async function processFramesSequentially(framesFile, totalFrames, tableName) {
  log(`Starting sequential processing of ${totalFrames} frames...`);
  
  // Start time tracking
  const startTime = Date.now();
  let successCount = 0;
  let failCount = 0;
  
  try {
    // Read all frames
    const allFrames = (await readFileAsync(framesFile, 'utf8')).trim().split('\n');
    
    if (allFrames.length === 0 || (allFrames.length === 1 && allFrames[0] === '')) {
      log('No frames to process');
      return { success: false, elapsed: 0 };
    }
    
    log(`Starting to process ${allFrames.length} frames sequentially...`);
    
    // Process each frame one at a time
    for (let i = 0; i < allFrames.length; i++) {
      const framePath = allFrames[i];
      const success = await processFrame(framePath, i, allFrames.length, tableName);
      
      if (success) {
        successCount++;
      } else {
        failCount++;
      }
      
      // Log progress every 10 frames or at the end
      if (i % 10 === 9 || i === allFrames.length - 1) {
        const elapsed = (Date.now() - startTime) / 1000;
        const framesPerSecond = (i + 1) / elapsed;
        const estimatedTimeRemaining = (allFrames.length - i - 1) / framesPerSecond;
        
        log(`Progress: ${i+1}/${allFrames.length} frames processed (${(((i+1)/allFrames.length)*100).toFixed(2)}%)`);
        log(`Success rate: ${successCount}/${i+1} (${((successCount/(i+1))*100).toFixed(2)}%)`);
        log(`Processing speed: ${framesPerSecond.toFixed(4)} frames/second`);
        log(`Elapsed time: ${formatTime(elapsed)}, Estimated time remaining: ${formatTime(estimatedTimeRemaining)}`);
      }
    }
    
    const elapsed = (Date.now() - startTime) / 1000;
    log(`Sequential processing completed in ${formatTime(elapsed)}`);
    log(`Final results: ${successCount} successful, ${failCount} failed`);
    
    return { 
      success: failCount === 0, 
      elapsed,
      successCount,
      failCount
    };
    
  } catch (error) {
    log(`Error in sequential processing: ${error.message}`);
    if (error.stack) {
      log(`Stack trace: ${error.stack}`);
    }
    return { success: false, elapsed: (Date.now() - startTime) / 1000 };
  }
}

// Format time in HH:MM:SS
function formatTime(seconds) {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Main execution function
async function main() {
  // Validate configuration first
  const { trackingTable, folderNameField, tableName } = validateConfig();
  
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
    
    // Process frames sequentially
    const result = await processFramesSequentially(framesFile, totalFrames, tableName);
    
    log(`Sequential processing complete: ${result.success ? 'All frames succeeded' : 'Some frames failed'}`);
    log(`Total processing time: ${formatTime(result.elapsed)}`);
    
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