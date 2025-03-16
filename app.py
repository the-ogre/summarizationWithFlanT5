import streamlit as st
import base64
import os
import torch
from tools import llm_pipeline, get_available_models
import traceback

st.set_page_config(
    page_title="LED Document Summarization App",
    page_icon="📄",
    layout="wide"
)

# Display GPU info if available
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device)
    gpu_info = f"GPU: {gpu_properties.name} ({gpu_properties.total_memory / 1e9:.2f} GB)"
    st.sidebar.info(f"🚀 {gpu_info}")
else:
    st.sidebar.warning("⚠️ No GPU detected. Processing will be slower.")

st.title("LED Document Summarization App")
st.markdown("Upload a PDF document and generate a concise summary using Longformer Encoder-Decoder model")
st.markdown("📚 *Using pszemraj/led-large-book-summary model - handles up to 16,384 tokens*")

# Sidebar for configuration
with st.sidebar:
    st.header("Model Configuration")
    model_name = st.selectbox(
        "Select Model",
        get_available_models(),
        index=0,
        help="Choose the model for summarization"
    )
    
    st.header("Summarization Settings")
    max_length = st.slider(
        "Maximum Summary Length",
        min_value=50,
        max_value=1024,
        value=256,
        step=32,
        help="Maximum length of the generated summary"
    )
    
    min_length = st.slider(
        "Minimum Summary Length",
        min_value=20,
        max_value=200,
        value=50,
        step=10,
        help="Minimum length of the generated summary"
    )
    
    chunk_size = st.slider(
        "Text Chunk Size",
        min_value=100,
        max_value=1000,
        value=200,
        step=50,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Overlap between consecutive chunks"
    )

@st.cache_data
def displayPDF(file_path):
    """Display a PDF file in the Streamlit app"""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')

        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        return pdf_display
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")
        return None

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Main area for file upload and display
uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

if uploaded_file is not None:
    # Save uploaded file
    filepath = os.path.join("data", uploaded_file.name)
    with open(filepath, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    
    # Display the PDF
    pdf_html = displayPDF(filepath)
    if pdf_html:
        st.markdown(pdf_html, unsafe_allow_html=True)
    
    # Summarize button
    if st.button("Summarize Document"):
        try:
            with st.spinner("Generating summary... This may take a few minutes depending on document length"):
                # Add a progress bar for better UX during long summarization
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress display
                status_text.text("Loading model... (this may take a moment)")
                progress_bar.progress(10)
                
                # Get summary
                summary = llm_pipeline(
                    filepath=filepath,
                    model_name=model_name,
                    max_length=max_length,
                    min_length=min_length,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Update progress to complete
                progress_bar.progress(100)
                status_text.empty()
                
                # Display summary
                st.subheader("Document Summary")
                st.success(summary)
                
                # Add option to download the summary
                summary_text = summary.encode()
                st.download_button(
                    label="Download Summary",
                    data=summary_text,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_summary.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"An error occurred during summarization: {str(e)}")
            st.error(traceback.format_exc())
else:
    st.info("👆 Upload a PDF document to start")

# Add footer
st.markdown("---")
st.markdown("Document Summarization App using Transformer models")