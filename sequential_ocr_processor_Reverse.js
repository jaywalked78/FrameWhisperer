const Airtable = require('airtable');
require('dotenv').config();
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configure Airtable
const BASE_ID = process.env.AIRTABLE_BASE_ID;
const TABLE_NAME = process.env.AIRTABLE_TABLE_NAME || 'tblFrameAnalysis';
const API_KEY = process.env.AIRTABLE_PERSONAL_ACCESS_TOKEN;

const base = new Airtable({ apiKey: API_KEY }).base(BASE_ID);
const table = base(TABLE_NAME);

// OCR processing configurations
const BATCH_SIZE = 10; // Process 10 frames at a time (Airtable limit)
const DELAY_BETWEEN_FRAMES = 0; // ms

// Log with timestamp
function log(message) {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
}

// Helper function for promises
function promisify(fn) {
  return function(...args) {
    return new Promise((resolve, reject) => {
      fn(...args, function(err, result) {
        if (err) return reject(err);
        resolve(result);
      });
    });
  };
}

// Sleep function
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Get all folders sorted chronologically
function getAllFolders(baseDir = '/home/jason/Videos/screenRecordings') {
  try {
    log(`Getting all folders in ${baseDir}...`);
    const stdout = execSync(`find "${baseDir}" -type d -name "screen_recording_*" | sort -r`).toString();
    const folders = stdout.trim().split('\n').filter(Boolean);
    log(`Found ${folders.length} folders`);
    return folders;
  } catch (error) {
    log(`Error getting folders: ${error.message}`);
    throw error;
  }
}

// Get records from Airtable by folder and filter out those that already have OCRData
// Returns records sorted by frame number
async function getRecordsWithoutOCR(folderPath) {
  return new Promise((resolve, reject) => {
    log(`Finding frames without OCR data in folder: ${folderPath}`);
    
    const records = [];
    const folderName = path.basename(folderPath);
    
    table.select({
      filterByFormula: `AND({FolderName} = '${folderName}', OR(NOT({OCRData}), {OCRData} = ""))`
    }).eachPage(
      function page(pageRecords, fetchNextPage) {
        records.push(...pageRecords);
        fetchNextPage();
      },
      function done(err) {
        if (err) {
          log(`Error fetching records: ${err.message}`);
          reject(err);
          return;
        }
        
        log(`Found ${records.length} frames without OCR data in folder ${folderName}`);
        
        // Sort records by frame number (extracted from FrameName or FolderPath)
        const sortedRecords = records.sort((a, b) => {
          // Extract frame number from the path or name
          const getFrameNumber = (record) => {
            const folderPath = record.get('FolderPath') || '';
            const frameName = path.basename(folderPath);
            // Extract numeric part from frameName (e.g. "frame_001.jpg" -> 1)
            const match = frameName.match(/(\d+)/);
            return match ? parseInt(match[1], 10) : 0;
          };
          
          return getFrameNumber(a) - getFrameNumber(b);
        });
        
        log(`Sorted ${sortedRecords.length} frames in numerical order`);
        resolve(sortedRecords);
      }
    );
  });
}

// Process a single frame with OCR and LLM
async function processFrame(record, folderPath) {
  const recordId = record.id;
  const framePath = record.get('FolderPath');
  
  if (!framePath) {
    log(`Record ${recordId} has no FolderPath, skipping`);
    return false;
  }
  
  log(`Processing frame: ${path.basename(framePath)}`);
  
  try {
    // Run OCR process on this specific frame
    const command = `python process_frames_by_path_Reverse.py --folder-path "${framePath}" --skip-airtable-update`;
    log(`Running command: ${command}`);
    
    const stdout = execSync(command).toString();
    log(`OCR Processing output: ${stdout.length > 500 ? stdout.substring(0, 500) + '...' : stdout}`);
    
    // Check if OCR result file exists
    const frameName = path.basename(framePath);
    const frameId = path.parse(frameName).name;
    const resultFile = path.join('output', 'ocr_results', `${frameId}.json`);
    
    if (!fs.existsSync(resultFile)) {
      log(`Error: OCR result file not found: ${resultFile}`);
      return false;
    }
    
    // Read OCR results
    const ocrResult = JSON.parse(fs.readFileSync(resultFile, 'utf8'));
    
    // Only update Airtable if we have valid results
    if (ocrResult && ocrResult.ocr_data) {
      const ocrData = ocrResult.ocr_data;
      const sensitiveFlag = ocrData.contains_sensitive_info;
      const filteredText = ocrData.ocr_text || 'No readable text';
      
      // Update Airtable manually
      log(`Updating Airtable record ${recordId} with OCR results`);
      
      // Set Flagged field to simple boolean string
      const sensitiveValue = sensitiveFlag ? 'true' : 'false';
      
      const updateFields = {
        'OCRData': filteredText,
        'Flagged': sensitiveValue
      };
      
      // Add sensitivity concerns if detected
      if (sensitiveFlag && ocrData.sensitive_content_types && ocrData.sensitive_content_types.length > 0) {
        updateFields['SensitivityConcerns'] = `Sensitive content detected: ${ocrData.sensitive_content_types.join(', ')}`;
      }
      
      await table.update(recordId, updateFields);
      log(`Successfully updated Airtable record for ${frameName}`);
      return true;
    } else {
      log(`No valid OCR data found in result file for ${frameName}`);
      return false;
    }
  } catch (error) {
    log(`Error processing frame ${framePath}: ${error.message}`);
    return false;
  }
}

// Process all frames in a folder sequentially
async function processFolder(folderPath) {
  try {
    log(`\n========== PROCESSING FOLDER: ${path.basename(folderPath)} ==========\n`);
    
    // Get records from Airtable without OCR data, sorted numerically
    const records = await getRecordsWithoutOCR(folderPath);
    
    if (records.length === 0) {
      log(`No frames without OCR data in folder ${path.basename(folderPath)}, skipping`);
      return 0;
    }
    
    log(`Found ${records.length} frames without OCR data to process in numerical order`);
    
    // Process each frame sequentially
    let successful = 0;
    let failed = 0;
    
    for (let i = 0; i < records.length; i++) {
      const record = records[i];
      const frameName = path.basename(record.get('FolderPath') || 'unknown');
      log(`Processing frame ${i+1}/${records.length}: ${frameName}`);
      
      const success = await processFrame(record, folderPath);
      
      if (success) {
        successful++;
      } else {
        failed++;
      }
      
      // Sleep between frames to avoid system overload
      await sleep(DELAY_BETWEEN_FRAMES);
    }
    
    log(`\n========== FOLDER SUMMARY: ${path.basename(folderPath)} ==========`);
    log(`Total frames: ${records.length}`);
    log(`Successfully processed: ${successful}`);
    log(`Failed: ${failed}`);
    log(`=================================================\n`);
    
    return successful;
  } catch (error) {
    log(`Error processing folder ${folderPath}: ${error.message}`);
    throw error;
  }
}

// Process a specific folder or all folders one by one
async function main() {
  try {
    // Get command line arguments
    const args = process.argv.slice(2);
    const folderArg = args.find(arg => arg.startsWith('--folder='));
    
    if (folderArg) {
      // Process specific folder
      const folderPath = folderArg.split('=')[1];
      log(`Processing specific folder: ${folderPath}`);
      await processFolder(folderPath);
    } else {
      // Process all folders chronologically
      log(`Processing all folders chronologically, one at a time`);
      const folders = getAllFolders();
      
      for (const folder of folders) {
        await processFolder(folder);
        // Additional sleep between folders
        await sleep(0);
      }
    }
    
    log(`OCR processing completed!`);
  } catch (error) {
    log(`Fatal error: ${error.message}`);
    process.exit(1);
  }
}

// Execute main function
main(); 