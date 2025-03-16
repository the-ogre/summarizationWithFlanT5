import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import (
    LEDTokenizer, 
    LEDForConditionalGeneration, 
    pipeline
)
import torch
import gc
import logging
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model location - can be overridden with environment variable
DEFAULT_MODEL_PATH = "models/led-large-book-summary"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

# Dictionary of available models with their paths
AVAILABLE_MODELS = {
    "LED-Large-Book-Summary": MODEL_PATH,
    # Add other models here as needed
}

def get_available_models() -> List[str]:
    """Return a list of available model names"""
    return list(AVAILABLE_MODELS.keys())

@st.cache_resource
def load_model(model_name: str) -> Tuple[Any, Any]:
    """
    Load model and tokenizer
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Tuple of (tokenizer, model)
    """
    try:
        model_path = AVAILABLE_MODELS.get(model_name, MODEL_PATH)
        logger.info(f"Loading model from {model_path}")
        
        # Clear GPU memory before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Cleared GPU memory. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Check if we're loading from local path or HuggingFace Hub
        if os.path.exists(model_path):
            tokenizer = LEDTokenizer.from_pretrained(model_path)
        else:
            # If model_path doesn't exist locally, assume it's a HuggingFace model ID
            logger.info(f"Model path not found locally, loading from HuggingFace: {model_path}")
            tokenizer = LEDTokenizer.from_pretrained("pszemraj/led-large-book-summary")
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model based on device
        if device == "cuda":
            # Try to load with CUDA optimizations
            try:
                # First try loading directly to GPU with half precision
                model = LEDForConditionalGeneration.from_pretrained(
                    "pszemraj/led-large-book-summary",
                    torch_dtype=torch.float16,  # Use float16 for GPU to save memory
                    low_cpu_mem_usage=True,
                ).to(device)
                logger.info("Model loaded to GPU with float16 precision")
            except Exception as e:
                logger.warning(f"Error loading model directly to GPU: {str(e)}")
                logger.warning("Falling back to loading model on CPU and then moving to GPU")
                
                # Fallback: load on CPU first then move to GPU
                model = LEDForConditionalGeneration.from_pretrained(
                    "pszemraj/led-large-book-summary",
                    torch_dtype=torch.float16
                )
                # Move model to GPU manually
                model = model.to(device)
                logger.info("Model loaded to CPU first, then moved to GPU")
            
            # Set to evaluation mode and disable gradient computation
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
        else:
            model = LEDForConditionalGeneration.from_pretrained(
                "pszemraj/led-large-book-summary",
                torch_dtype=torch.float32
            )
            logger.info("Model loaded to CPU with float32 precision")
            
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def file_preprocessing(file_path: str, chunk_size: int = 200, chunk_overlap: int = 50) -> str:
    """
    Process PDF file and extract text
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        Extracted text from the PDF
    """
    try:
        logger.info(f"Processing file: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        logger.info(f"Loaded {len(pages)} pages")
        
        # Show progress in Streamlit
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        texts = text_splitter.split_documents(pages)
        logger.info(f"Split into {len(texts)} chunks")
        
        final_texts = ""
        for i, text in enumerate(texts):
            final_texts += text.page_content + " "
            
            # Update progress
            progress = (i + 1) / len(texts)
            progress_bar.progress(progress)
            status_text.text(f"Processing text chunk {i+1}/{len(texts)}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return final_texts
    except Exception as e:
        logger.error(f"Error in file preprocessing: {str(e)}")
        raise

def llm_pipeline(
    filepath: str,
    model_name: str = "LED-Large-Book-Summary",
    max_length: int = 512,
    min_length: int = 50,
    chunk_size: int = 200,
    chunk_overlap: int = 50
) -> str:
    """
    Process a document and generate a summary using LED model
    
    Args:
        filepath: Path to the PDF file
        model_name: Name of the model to use
        max_length: Maximum length of the summary
        min_length: Minimum length of the summary
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between consecutive chunks
        
    Returns:
        Generated summary
    """
    try:
        # Load model and tokenizer
        tokenizer, base_model = load_model(model_name)
        
        # Get device
        device = next(base_model.parameters()).device
        logger.info(f"Model is on device: {device}")
        
        # Create summarization pipeline with device specified
        summarizer = pipeline(
            'summarization',
            model=base_model,
            tokenizer=tokenizer,
            device=0 if str(device).startswith("cuda") else -1,
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            repetition_penalty=3.5,
            num_beams=4,
            early_stopping=True
        )
        
        # Process the file
        input_text = file_preprocessing(
            filepath, 
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Check input text length
        input_tokens = len(tokenizer.encode(input_text))
        logger.info(f"Input text length: {input_tokens} tokens")
        
        # If input is too long, truncate to LED max length
        max_input_length = 16384  # LED model can handle up to 16,384 tokens
        
        if input_tokens > max_input_length:
            st.warning(f"Document is very large ({input_tokens} tokens). Truncating to fit model capacity of 16,384 tokens.")
            input_text = tokenizer.decode(tokenizer.encode(input_text)[:max_input_length])
        
        # Generate summary
        logger.info("Generating summary")
        with st.spinner("Generating summary with LED model (this may take a few minutes)..."):
            result = summarizer(input_text)
            summary = result[0]['summary_text']
        
        return summary
    except Exception as e:
        logger.error(f"Error in summarization pipeline: {str(e)}")
        st.error(f"Error in summarization: {str(e)}")
        raise