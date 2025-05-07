#!/usr/bin/env python3
"""
Test script to demonstrate the improved OCR text formatting
"""

import os
import json
import sys
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

# Load environment variables
load_dotenv()

# Set up argument parser
parser = argparse.ArgumentParser(description="Test OCR text formatting with Gemini")
parser.add_argument("--image", required=True, help="Path to the image file to test")
args = parser.parse_args()

# Configure Gemini API
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    sys.exit(1)

genai.configure(api_key=api_key)

# Initialize the model
model_name = os.environ.get("GEMINI_PREFERRED_MODEL", "models/gemini-2.0-flash")
model = genai.GenerativeModel(model_name)
print(f"Using model: {model_name}")

# Load the image
if not os.path.exists(args.image):
    print(f"Error: Image file not found: {args.image}")
    sys.exit(1)

try:
    img = Image.open(args.image)
    print(f"Loaded image: {args.image} - Size: {img.size}")
except Exception as e:
    print(f"Error loading image: {str(e)}")
    sys.exit(1)

# Prepare the prompt
prompt = """
Analyze this screen capture.

Your task:
1. Extract all meaningful text visible in the image
2. You MUST try to identify words and labels even if the text is incomplete or garbled
3. If there are menus, buttons, or UI elements with text, include them
4. Remove errors and random artifacts
5. Flag any sensitive information (passwords, API keys, credentials, personal data)
6. IMPORTANT: NEVER return "No readable text" unless the image truly has NO text at all
7. Even partial or single words are better than returning "No readable text"
8. Organize text with category labels like "UI:", "Content:", "Menu:" etc. followed by text
9. Use pipe symbols (|) to separate different UI elements and sections

RESPONSE FORMAT:
You must respond EXCLUSIVELY with a valid JSON object that follows this exact schema:
{
    "filtered_text": string,  // ALL text found in the image, formatted with sections and separators
    "contains_sensitive_info": boolean,  // Must be exactly true or false
    "sensitive_content_types": string[]  // Array of strings, empty if no sensitive info
}

EXAMPLES:

Example 1 (UI elements):
{
    "filtered_text": "App: Firefox | Time: 3:45 PM | UI: Home | Dashboard | Settings | Profile | Logout | Section: Search... | Content: Recent Activity: No new notifications",
    "contains_sensitive_info": false,
    "sensitive_content_types": []
}

Example 2 (With sensitive info):
{
    "filtered_text": "Page: Settings | Section: API Configuration | Key: sk_test_EXAMPLE_KEY_PLACEHOLDER | User: admin@example.com",
    "contains_sensitive_info": true,
    "sensitive_content_types": ["api_key", "email"]
}

DO NOT include any explanations, markdown formatting, or code blocks - JUST THE JSON OBJECT.
"""

# Process the image
print("Processing image with Gemini...")
try:
    response = model.generate_content([prompt, img])
    
    # Handle different response formats
    text = response.text
    if "```json" in text:
        json_str = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        json_str = text.split("```")[1].strip()
    else:
        json_str = text.strip()
    
    # Parse the JSON response
    result = json.loads(json_str)
    
    # Display the results
    print("\n=== RESULTS ===\n")
    print("Contains sensitive info:", result.get("contains_sensitive_info", False))
    if result.get("contains_sensitive_info"):
        print("Sensitive content types:", ", ".join(result.get("sensitive_content_types", [])))
    
    print("\nExtracted Text (Formatted):")
    print("--------------------------")
    print(result.get("filtered_text", "No text extracted"))
    
    # Show formatted elements
    if "|" in result.get("filtered_text", ""):
        print("\nSeparate Elements:")
        print("-----------------")
        elements = result.get("filtered_text", "").split("|")
        for i, element in enumerate(elements, 1):
            print(f"{i}. {element.strip()}")
    
except Exception as e:
    print(f"Error processing image: {str(e)}")
    sys.exit(1) 