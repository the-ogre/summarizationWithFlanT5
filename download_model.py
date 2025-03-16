#!/usr/bin/env python3
"""
Script to download models for the Document Summarization App.
This allows users to easily download the needed models without manual intervention.
"""

import os
import argparse
from transformers import LEDTokenizer, LEDForConditionalGeneration

def download_model(model_name, output_dir):
    """
    Download a model from Hugging Face and save it to the specified directory.
    
    Args:
        model_name: Name of the model on Hugging Face Hub
        output_dir: Directory to save the model
    """
    print(f"Downloading model {model_name} to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download tokenizer and model
    tokenizer = LEDTokenizer.from_pretrained(model_name)
    model = LEDForConditionalGeneration.from_pretrained(model_name)
    
    # Save to specified directory
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    
    print(f"Model successfully downloaded to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download models for Document Summarization App")
    parser.add_argument(
        "--model", 
        default="pszemraj/led-large-book-summary", 
        help="Model name from Hugging Face Hub (default: pszemraj/led-large-book-summary)"
    )
    parser.add_argument(
        "--output", 
        default="models/led-large-book-summary", 
        help="Output directory for the model files"
    )
    
    args = parser.parse_args()
    download_model(args.model, args.output)

if __name__ == "__main__":
    main()