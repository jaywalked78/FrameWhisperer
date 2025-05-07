#!/usr/bin/env python3
# gemini_processor.py - Gemini AI integration for OCR Pipeline
# This module provides a standardized interface for using Gemini API in OCR processing

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
from pydantic import BaseModel, Field, validator
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gemini_processor")

# Load environment variables
load_dotenv()

class ModelGeneration(str, Enum):
    """Gemini model generations"""
    GEMINI_1_0 = "1.0"
    GEMINI_1_5 = "1.5"
    GEMINI_2_0 = "2.0" 
    GEMINI_2_5 = "2.5"

class ModelTier(str, Enum):
    """Gemini model tiers"""
    FLASH = "flash"
    PRO = "pro"
    VISION = "vision"
    EMBEDDING = "embedding"

class GeminiModelConfig(BaseModel):
    """Configuration for a Gemini model"""
    name: str
    display_name: Optional[str] = None
    generation: Optional[ModelGeneration] = None
    tier: Optional[ModelTier] = None
    is_experimental: bool = False
    is_preview: bool = False
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None
    supports_vision: bool = False
    temperature: float = 0.4
    top_p: float = 1.0
    top_k: int = 32
    max_output_tokens: Optional[int] = None
    
    @validator('max_output_tokens', always=True)
    def set_default_max_output(cls, v, values):
        """Set default max_output_tokens based on output_token_limit"""
        if v is None and values.get('output_token_limit'):
            return min(values['output_token_limit'], 8192)  # Default to 8k or limit, whichever is smaller
        return v

class ProcessingResult(BaseModel):
    """Result of a Gemini processing operation"""
    success: bool
    content: Optional[str] = None
    error: Optional[str] = None
    usage: Dict[str, int] = Field(default_factory=dict)
    processing_time: float = 0.0
    model_used: Optional[str] = None

class GeminiProcessor:
    """Processor class for Gemini AI operations in OCR Pipeline"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        default_model: str = "models/gemini-1.5-flash",
        fallback_models: List[str] = None,
        allow_experimental: bool = False,
        temperature: float = 0.4,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """Initialize the Gemini processor with configuration
        
        Args:
            api_key: Gemini API key (uses GOOGLE_API_KEY env if not provided)
            default_model: Default model to use for text processing
            fallback_models: List of models to try if default fails
            allow_experimental: Whether to allow experimental models
            temperature: Default temperature for generation
            max_retries: Maximum number of retries on error
            retry_delay: Delay between retries in seconds
        """
        # Get API key from provided value or environment
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter")
            
        # Configure Gemini API
        genai.configure(
            api_key=self.api_key,
            transport="rest",
            client_options={"api_endpoint": "generativelanguage.googleapis.com"}
        )
        
        # Set default parameters
        self.default_model = default_model
        self.fallback_models = fallback_models or [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro",
            "models/gemini-1.0-pro"
        ]
        self.allow_experimental = allow_experimental
        self.default_temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Cache for available models
        self._models_cache = None
        self._models_cache_time = 0
        self._cache_valid_duration = 3600  # 1 hour
        
        # Initialize model configurations
        self.model_configs = {}
        
        logger.info(f"Initialized GeminiProcessor with default model: {default_model}")
    
    def _is_cache_valid(self) -> bool:
        """Check if models cache is still valid"""
        return (
            self._models_cache is not None and 
            time.time() - self._models_cache_time < self._cache_valid_duration
        )
    
    def list_available_models(self, force_refresh: bool = False) -> List[GeminiModelConfig]:
        """List all available Gemini models
        
        Args:
            force_refresh: Whether to force a refresh of the cache
            
        Returns:
            List of model configurations
        """
        if not self._is_cache_valid() or force_refresh:
            try:
                # Get models from API
                models = genai.list_models()
                
                # Filter for Gemini models
                gemini_models = [m for m in models if "gemini" in m.name.lower()]
                
                # Parse into model configurations
                model_configs = []
                
                for model in gemini_models:
                    # Extract model name without path
                    model_name_short = model.name.split('/')[-1] if '/' in model.name else model.name
                    
                    # Extract generation
                    generation = None
                    if "gemini-" in model_name_short:
                        parts = model_name_short.split("-")
                        if len(parts) > 1 and parts[1] in [g.value for g in ModelGeneration]:
                            generation = parts[1]
                    
                    # Extract tier
                    tier = None
                    for t in ModelTier:
                        if t.value in model_name_short.lower():
                            tier = t.value
                            break
                    
                    # Determine if experimental or preview
                    is_experimental = "experimental" in model.name.lower() or "exp" in model.name.lower()
                    is_preview = "preview" in model.name.lower()
                    
                    # Determine vision capabilities
                    supports_vision = "vision" in model_name_short.lower()
                    
                    # Create model config
                    config = GeminiModelConfig(
                        name=model.name,
                        display_name=getattr(model, "display_name", None),
                        generation=generation,
                        tier=tier,
                        is_experimental=is_experimental,
                        is_preview=is_preview,
                        input_token_limit=getattr(model, "input_token_limit", None),
                        output_token_limit=getattr(model, "output_token_limit", None),
                        supports_vision=supports_vision
                    )
                    
                    model_configs.append(config)
                    # Also store in the instance cache
                    self.model_configs[model.name] = config
                
                # Update cache
                self._models_cache = model_configs
                self._models_cache_time = time.time()
                
                logger.info(f"Retrieved {len(model_configs)} Gemini models")
                return model_configs
                
            except Exception as e:
                logger.error(f"Error listing models: {str(e)}")
                # If cache exists, use it despite error
                if self._models_cache:
                    logger.info("Using cached models due to API error")
                    return self._models_cache
                # Otherwise return empty list
                return []
        
        # Return from cache
        return self._models_cache
    
    def get_model_info(self, model_name: str) -> Optional[GeminiModelConfig]:
        """Get information about a specific model
        
        Args:
            model_name: Name of the model to get information for
            
        Returns:
            Model configuration or None if not found
        """
        # If the model info is already cached, return it
        if model_name in self.model_configs:
            return self.model_configs[model_name]
        
        # Otherwise, ensure we have loaded models
        if not self._is_cache_valid():
            self.list_available_models()
            
        # Try to find the model in the cache
        if model_name in self.model_configs:
            return self.model_configs[model_name]
            
        # If still not found, try with the normalized name
        # (sometimes models are referenced with or without 'models/' prefix)
        normalized_name = model_name
        if not normalized_name.startswith("models/"):
            normalized_name = f"models/{normalized_name}"
        
        if normalized_name in self.model_configs:
            return self.model_configs[normalized_name]
            
        # If we still can't find it, return None
        return None
    
    def find_best_model(
        self, 
        generation: Optional[ModelGeneration] = ModelGeneration.GEMINI_1_5,
        tier: Optional[ModelTier] = ModelTier.FLASH,
        supports_vision: bool = False,
        allow_experimental: Optional[bool] = None,
        allow_preview: bool = True
    ) -> Optional[str]:
        """Find the best available model matching criteria
        
        Args:
            generation: Preferred generation (1.0, 1.5, 2.0, 2.5)
            tier: Preferred tier (flash, pro, vision)
            supports_vision: Whether vision capabilities are required
            allow_experimental: Whether to allow experimental models
            allow_preview: Whether to allow preview models
            
        Returns:
            Name of best matching model or None if no match
        """
        # Use instance setting if not specified
        if allow_experimental is None:
            allow_experimental = self.allow_experimental
            
        # Get available models
        models = self.list_available_models()
        
        # Filter based on criteria
        filtered_models = []
        for model in models:
            # Skip models not matching required criteria
            if supports_vision and not model.supports_vision:
                continue
                
            if model.is_experimental and not allow_experimental:
                continue
                
            if model.is_preview and not allow_preview:
                continue
            
            # Calculate match score (higher is better)
            score = 0
            
            # Exact generation and tier match is highest priority
            if model.generation == generation and model.tier == tier:
                score += 100
            # Generation match
            elif model.generation == generation:
                score += 50
            # Tier match
            elif model.tier == tier:
                score += 25
                
            # Stable models preferred over preview or experimental
            if not model.is_experimental and not model.is_preview:
                score += 10
            elif not model.is_experimental and model.is_preview:
                score += 5
                
            # Add to candidates with score
            filtered_models.append((model, score))
        
        # Sort by score (highest first)
        filtered_models.sort(key=lambda x: x[1], reverse=True)
        
        if filtered_models:
            best_model = filtered_models[0][0]
            logger.info(f"Selected best model: {best_model.name}")
            return best_model.name
            
        # No matching model found
        logger.warning(f"No matching model found for criteria: {generation}, {tier}, vision={supports_vision}")
        return None
    
    def process_text(
        self, 
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: float = 1.0,
        top_k: int = 32,
        include_safety_attributes: bool = True
    ) -> ProcessingResult:
        """Process text with Gemini
        
        Args:
            prompt: The prompt text to process
            model: Model to use (defaults to self.default_model)
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
            top_p: Top p sampling parameter
            top_k: Top k sampling parameter
            include_safety_attributes: Whether to include safety attributes
            
        Returns:
            ProcessingResult with success status and content
        """
        start_time = time.time()
        
        # Use defaults if not specified
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.default_temperature
        
        models_to_try = [model]
        # Add fallback models if primary is not default
        if model != self.default_model:
            models_to_try.extend([m for m in self.fallback_models if m != model])
        # Add default model as last resort if not already included
        if self.default_model not in models_to_try:
            models_to_try.append(self.default_model)
            
        last_error = None
        
        # Try each model
        for model_name in models_to_try:
            logger.info(f"Attempting processing with model: {model_name}")
            
            for attempt in range(self.max_retries):
                try:
                    # Configure model
                    model_instance = genai.GenerativeModel(
                        model_name,
                        generation_config={
                            "temperature": temperature,
                            "top_p": top_p,
                            "top_k": top_k,
                            "max_output_tokens": max_output_tokens,
                        }
                    )
                    
                    # Generate content
                    response = model_instance.generate_content(
                        prompt,
                        safety_settings=None,  # Use default safety settings
                    )
                    
                    # Extract usage information if available
                    usage = {}
                    if hasattr(response, "usage_metadata"):
                        metadata = response.usage_metadata
                        usage = {
                            "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                            "candidates_token_count": getattr(metadata, "candidates_token_count", 0),
                            "total_token_count": getattr(metadata, "total_token_count", 0),
                        }
                    
                    # Calculate processing time
                    processing_time = time.time() - start_time
                    
                    # Return successful result
                    return ProcessingResult(
                        success=True,
                        content=response.text,
                        error=None,
                        usage=usage,
                        processing_time=processing_time,
                        model_used=model_name
                    )
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Error (attempt {attempt+1}/{self.max_retries}) with {model_name}: {last_error}")
                    
                    # Specific handling for certain errors
                    if "API_KEY_INVALID" in last_error or "PERMISSION_DENIED" in last_error:
                        # No point retrying the same model with an invalid key or permission issue
                        break
                        
                    # Delay before retry
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
        
        # If we get here, all models failed
        processing_time = time.time() - start_time
        logger.error(f"All models failed. Last error: {last_error}")
        
        return ProcessingResult(
            success=False,
            content=None,
            error=last_error,
            usage={},
            processing_time=processing_time,
            model_used=model
        )
    
    def process_vision(
        self,
        prompt: str,
        image_data: Union[str, bytes],
        image_format: str = "png",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ProcessingResult:
        """Process text and image with Gemini Vision capabilities
        
        Args:
            prompt: The prompt text
            image_data: Image data as base64 string or bytes
            image_format: Image format (png, jpeg, etc.)
            model: Model to use (defaults to vision-capable model)
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
            
        Returns:
            ProcessingResult with success status and content
        """
        start_time = time.time()
        
        # Find a vision-capable model if none specified
        if model is None:
            model = self.find_best_model(
                generation=ModelGeneration.GEMINI_1_5,
                tier=ModelTier.PRO,
                supports_vision=True
            )
            if not model:
                # Fall back to known vision models
                model = "models/gemini-1.5-pro"
        
        # Use defaults if not specified
        temperature = temperature if temperature is not None else self.default_temperature
        
        try:
            # Configure model
            model_instance = genai.GenerativeModel(
                model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
            )
            
            # Prepare image data
            if isinstance(image_data, str):
                # Assume it's already a base64 string or file path
                if os.path.exists(image_data):
                    # It's a file path
                    with open(image_data, "rb") as f:
                        image_bytes = f.read()
                else:
                    # It's a base64 string
                    import base64
                    image_bytes = base64.b64decode(image_data)
            else:
                # It's already bytes
                image_bytes = image_data
            
            # Prepare multipart content
            multipart_content = [
                prompt,  # Text part
                {"mime_type": f"image/{image_format}", "data": image_bytes}  # Image part
            ]
            
            # Generate content
            response = model_instance.generate_content(multipart_content)
            
            # Extract usage information
            usage = {}
            if hasattr(response, "usage_metadata"):
                metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                    "candidates_token_count": getattr(metadata, "candidates_token_count", 0),
                    "total_token_count": getattr(metadata, "total_token_count", 0),
                }
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log raw response for debugging
            raw_text = response.text if hasattr(response, 'text') else str(response)
            logger.info(f"Raw vision response (first 100 chars): {raw_text[:100]}...")
            
            # Check for empty response
            if not raw_text or raw_text.strip() == "":
                logger.error("Empty response from vision model")
                return ProcessingResult(
                    success=False,
                    content="",
                    error="Empty response from vision model",
                    usage=usage,
                    processing_time=processing_time,
                    model_used=model
                )
            
            # Return successful result
            return ProcessingResult(
                success=True,
                content=raw_text,
                error=None,
                usage=usage,
                processing_time=processing_time,
                model_used=model
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Vision processing error: {error_msg}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=False,
                content=None,
                error=error_msg,
                usage={},
                processing_time=processing_time,
                model_used=model
            )
    
    def get_token_count(self, text: str, model: Optional[str] = None) -> int:
        """Get token count for a text string
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting (defaults to default_model)
            
        Returns:
            Token count or 0 if counting fails
        """
        model = model or self.default_model
        
        try:
            model_instance = genai.GenerativeModel(model)
            result = model_instance.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            # Fallback rough estimation (3 tokens per word)
            word_count = len(text.split())
            return word_count * 3
    
    def is_api_healthy(self) -> bool:
        """Check if the API is functioning correctly
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            # Try to list models as a basic health check
            models = genai.list_models()
            return len(models) > 0
        except Exception:
            return False
            
    def extract_structure_from_text(
        self,
        text: str,
        structure_format: str = "json",
        structure_description: str = "",
        model: Optional[str] = None,
    ) -> Tuple[bool, Any]:
        """Extract structured data from unstructured text
        
        Args:
            text: Text to extract structure from
            structure_format: Format to extract (json, xml, etc.)
            structure_description: Description of the structure to extract
            model: Model to use (defaults to default_model)
            
        Returns:
            Tuple of (success, extracted_data)
        """
        # Construct a prompt that asks for structured data
        struct_prompt = f"""Extract structured {structure_format} from the following text.
        
{structure_description}

Output ONLY valid {structure_format}, no explanations or other text.

TEXT TO EXTRACT FROM:
{text}
"""
        
        # Process with Gemini
        result = self.process_text(
            prompt=struct_prompt,
            model=model,
            temperature=0.1,  # Low temperature for more deterministic output
        )
        
        if not result.success:
            return False, None
            
        # Try to parse the result based on format
        try:
            if structure_format.lower() == "json":
                # Find JSON in the text
                import re
                json_match = re.search(r'({[\s\S]*}|\[[\s\S]*\])', result.content)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                else:
                    data = json.loads(result.content)
                return True, data
            elif structure_format.lower() == "xml":
                # Return as string, let caller parse
                return True, result.content
            else:
                # For other formats, return raw text
                return True, result.content
        except Exception as e:
            logger.error(f"Error parsing {structure_format}: {str(e)}")
            # Return the raw text in case caller wants to handle it
            return False, result.content

    def process_image(self, image_path, prompt):
        """
        Process an image with the Gemini model.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt to send with the image
            
        Returns:
            ProcessingResult object with the result
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return ProcessingResult(
                    success=False,
                    error=f"Image file not found: {image_path}",
                    model_used=self.default_model
                )
            
            # Add JSON formatting instructions to the prompt
            enhanced_prompt = prompt
            
            # Ensure the prompt includes explicit JSON formatting instructions
            if "JSON" not in prompt:
                enhanced_prompt = prompt + "\n\nIMPORTANT: Format your response as a valid JSON object that includes 'filtered_text', 'contains_sensitive_info', and 'sensitive_content_types' fields."
            
            # Make sure the prompt includes a warning about formatting
            if "DO NOT include any explanations" not in prompt:
                enhanced_prompt += "\n\nDO NOT include any explanations, markdown formatting, or code blocks - JUST THE JSON OBJECT."
            
            logger.info(f"Enhanced prompt with JSON formatting instructions")
            
            # Directly try the Gemini 2.0 Flash Exp model which supports vision
            vision_model = "models/gemini-2.0-flash-exp"
            logger.info(f"Using Gemini 2.0 Flash Exp for vision: {vision_model}")
            
            # If that fails, we'll try these backup models in order
            backup_models = [
                "models/gemini-2.0-flash", 
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro-vision", 
                "models/gemini-1.5-pro"
            ]
            
            try:
                # First try with the primary vision model
                result = self.process_vision(
                    prompt=enhanced_prompt,
                    image_data=image_path,
                    image_format="jpg",
                    model=vision_model
                )
                
                # If successful, return the result
                if result.success:
                    # Try to extract JSON from the response if needed
                    content = result.content
                    # Check if the response contains JSON object markers
                    if "{" in content and "}" in content:
                        import re
                        json_match = re.search(r'({[\s\S]*})', content)
                        if json_match:
                            # Extract just the JSON part
                            json_str = json_match.group(1)
                            logger.info(f"Extracted JSON from response: {json_str[:100]}...")
                            # Create a new result with the extracted JSON
                            return ProcessingResult(
                                success=True,
                                content=json_str,
                                error=None,
                                usage=result.usage,
                                processing_time=result.processing_time,
                                model_used=result.model_used
                            )
                    
                    return result
                    
                # If not successful, log the error and continue to fallbacks
                logger.warning(f"Primary vision model {vision_model} failed: {result.error}")
                
                # Try backup models
                for model in backup_models:
                    logger.info(f"Trying backup vision model: {model}")
                    try:
                        result = self.process_vision(
                            prompt=enhanced_prompt,
                            image_data=image_path,
                            image_format="jpg",
                            model=model
                        )
                        if result.success:
                            logger.info(f"Successfully processed with backup model: {model}")
                            
                            # Try to extract JSON from the response if needed
                            content = result.content
                            # Check if the response contains JSON object markers
                            if "{" in content and "}" in content:
                                import re
                                json_match = re.search(r'({[\s\S]*})', content)
                                if json_match:
                                    # Extract just the JSON part
                                    json_str = json_match.group(1)
                                    logger.info(f"Extracted JSON from response: {json_str[:100]}...")
                                    # Create a new result with the extracted JSON
                                    return ProcessingResult(
                                        success=True,
                                        content=json_str,
                                        error=None,
                                        usage=result.usage,
                                        processing_time=result.processing_time,
                                        model_used=result.model_used
                                    )
                            
                            return result
                        else:
                            logger.warning(f"Backup model {model} failed: {result.error}")
                    except Exception as e:
                        logger.warning(f"Error with backup model {model}: {str(e)}")
                
                # If we get here, all models failed
                logger.error("All vision models failed")
                return ProcessingResult(
                    success=False,
                    error="All vision models failed",
                    model_used=vision_model
                )
                
            except Exception as model_error:
                logger.error(f"Error using vision models: {str(model_error)}")
                return ProcessingResult(
                    success=False,
                    error=str(model_error),
                    model_used=vision_model
                )
            
        except Exception as e:
            # Return a failure result with the error
            return ProcessingResult(
                success=False,
                error=str(e),
                model_used=self.default_model
            )

# Simple usage example
if __name__ == "__main__":
    processor = GeminiProcessor()
    print(f"API Healthy: {processor.is_api_healthy()}")
    
    models = processor.list_available_models()
    print(f"Found {len(models)} Gemini models")
    
    # Test processing
    result = processor.process_text("Explain what OCR is in one short paragraph.")
    if result.success:
        print(f"Response: {result.content}")
        print(f"Tokens used: {result.usage}")
        print(f"Processing time: {result.processing_time:.2f}s")
    else:
        print(f"Error: {result.error}") 