#!/usr/bin/env python3
# gemini_2_0_flash_example.py - Example using Gemini 2.0 Flash models with our processor

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from gemini_processor import GeminiProcessor, ModelGeneration, ModelTier, ProcessingResult

# Load environment variables
load_dotenv()

def test_model(processor: GeminiProcessor, model_name: str, prompt: str) -> None:
    """Test a specific model with the given prompt"""
    print(f"\n=== TESTING MODEL: {model_name} ===")
    start_time = time.time()
    
    result = processor.process_text(
        prompt=prompt,
        model=model_name,
        temperature=0.2  # Lower temperature for more deterministic output
    )
    
    duration = time.time() - start_time
    
    if result.success:
        print(f"✅ Success! ({duration:.2f}s)")
        print(f"Response:\n{result.content}")
        if result.usage:
            print(f"Tokens: {result.usage.get('total_token_count', 'N/A')}")
    else:
        print(f"❌ Failed: {result.error}")

def find_gemini_2_flash_models(processor: GeminiProcessor) -> Dict[str, List[str]]:
    """Find all available Gemini 2.0 Flash models, grouped by type"""
    models = processor.list_available_models(force_refresh=True)
    
    # Filter for 2.0 Flash models
    flash_models = {
        "stable": [],
        "experimental": [],
        "preview": []
    }
    
    for model_config in models:
        if (model_config.generation == "2.0" and 
            "flash" in model_config.name.lower() and
            "thinking" not in model_config.name.lower()):  # Exclude thinking models
            
            if model_config.is_experimental:
                flash_models["experimental"].append(model_config.name)
            elif model_config.is_preview:
                flash_models["preview"].append(model_config.name)
            else:
                flash_models["stable"].append(model_config.name)
    
    # Print available models
    print(f"Found {len(flash_models['stable'] + flash_models['experimental'] + flash_models['preview'])} Gemini 2.0 Flash models:")
    
    for category, models in flash_models.items():
        if models:
            print(f"\n{category.upper()} MODELS:")
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")
    
    return flash_models

def compare_models(processor: GeminiProcessor, model_names: List[str], prompt: str) -> None:
    """Compare multiple models using the same prompt"""
    results = []
    
    print(f"\n=== COMPARING {len(model_names)} MODELS ===")
    print(f"Prompt: {prompt}")
    
    for model_name in model_names:
        start_time = time.time()
        result = processor.process_text(prompt=prompt, model=model_name)
        duration = time.time() - start_time
        
        model_result = {
            "model": model_name,
            "success": result.success,
            "duration": duration,
            "tokens": result.usage.get("total_token_count", 0) if result.success else 0,
            "error": result.error if not result.success else None,
            "content": result.content if result.success else None
        }
        results.append(model_result)
    
    # Sort by success then speed
    results.sort(key=lambda x: (-int(x["success"]), x["duration"]))
    
    # Print comparison table
    print("\n=== RESULTS (sorted by success and speed) ===")
    print(f"{'MODEL':<40} | {'SUCCESS':<7} | {'TIME':<6} | {'TOKENS':<6}")
    print("-" * 65)
    
    for result in results:
        model_short = result["model"].split("/")[-1]
        success_str = "✅" if result["success"] else "❌"
        print(f"{model_short:<40} | {success_str:<7} | {result['duration']:.2f}s | {result['tokens']:<6}")

def main():
    # Initialize the processor with support for experimental models
    processor = GeminiProcessor(
        default_model="models/gemini-2.0-flash",  # Default to 2.0 model
        allow_experimental=True,  # Allow experimental models
        max_retries=2,
        temperature=0.2
    )
    
    # Find all Gemini 2.0 Flash models
    flash_models = find_gemini_2_flash_models(processor)
    
    # Define test prompts
    ocr_prompt = """
    Analyze the following extracted OCR text and format it into a properly structured document:
    
    INVOICE
    InvoiceNo: INV-2023-456
    DateL: 2023-05-15
    
    Bill To:
    Acme corporation 
    123 main Street
    Anytown, AT 12345
    
    Items:
    1. consulting Services - 10 h0urs @ $150/hr: $1,500.00
    2. softwate Licanse - 5 users @ $99/user: $495.00
    3. Equipment rental - 1 month: $750.00
    
    Subtotal: $2,745.00
    Tax (8%): $219,600
    Total Due: $2,964.60
    
    Payment Tarms: Net 30 days
    Please make checks payable to: Tech Solutions Inc.
    """
    
    analysis_prompt = "What are the key differences between Gemini 1.5 and Gemini 2.0 models in terms of capabilities and performance?"
    
    # Select specific models to test
    # Try to test at least one stable and one experimental 2.0 Flash model if available
    models_to_test = []
    
    if flash_models["stable"]:
        models_to_test.append(flash_models["stable"][0])  # First stable model
        
    if flash_models["experimental"]:
        models_to_test.append(flash_models["experimental"][0])  # First experimental model
    
    # Add fallback to 1.5 for comparison
    models_to_test.append("models/gemini-1.5-flash")
    
    # Run comparison on selected models
    compare_models(processor, models_to_test, ocr_prompt)
    
    # Test the best 2.0 Flash experimental model if available
    if flash_models["experimental"]:
        exp_model = flash_models["experimental"][0]
        test_model(processor, exp_model, analysis_prompt)

if __name__ == "__main__":
    main() 