# LED Document Summarization App

A Streamlit application that summarizes documents using the Longformer Encoder-Decoder (LED) model, specifically the "pszemraj/led-large-book-summary" model which is fine-tuned for book summarization.

## Features

- PDF document upload and display
- Leverages the LED model to handle long documents (up to 16,384 tokens)
- Optimized for book and long-form content summarization
- Configurable summarization parameters
- Progress indicators during processing
- Error handling and logging
- Summary download option
- Responsive UI with sidebar configuration

## Requirements

- Python 3.11 or later
- PyTorch
- Transformers
- LangChain
- Streamlit
- See `requirements.txt` for complete dependencies

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/document-summarization-app.git
cd document-summarization-app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the LED model:
```bash
mkdir -p models
python download_model.py

# Or use the model directly from HuggingFace by setting
# MODEL_PATH=pszemraj/led-large-book-summary in your .env file
```

5. Configure the application:
```bash
cp .env.example .env
# Edit .env file to match your environment
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser at `http://localhost:8501`

3. Upload a PDF document and configure summarization parameters in the sidebar

4. Click "Summarize Document" to generate a summary

## Model Configuration

The application is configured to use the LED Large Book Summary model by default. This model:

- Handles up to 16,384 tokens of input
- Is fine-tuned on the BookSum dataset
- Is optimized for long-form content like books, articles, and reports

To add or change models:

1. Add your model to the `models` directory or specify its path in the `.env` file
2. Update the `AVAILABLE_MODELS` dictionary in `tools.py` with your new model

### Optimization Notes

For best results with the LED model, the application uses these parameters:
- `no_repeat_ngram_size=3`
- `encoder_no_repeat_ngram_size=3`
- `repetition_penalty=3.5`
- `num_beams=4`
- `early_stopping=True`

These settings help create more coherent and less repetitive summaries.

## Project Structure

- `app.py`: Main Streamlit application
- `tools.py`: Model loading and text processing utilities
- `requirements.txt`: Package dependencies
- `Dockerfile`: Container configuration
- `.env.example`: Example environment configuration
- `data/`: Directory for uploaded documents (created automatically)
- `models/`: Directory for model files

## License

[MIT License](LICENSE)