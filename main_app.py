import os
import io
import pickle
import requests

import numpy as np
import faiss
import pytesseract
import streamlit as st

from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer

# Initial Streamlit Page Configuration
st.set_page_config(page_title="AI-Powered Offline PDF Q&A", layout="wide")

# --- Constants for File Paths --- #
TEXT_LABELS_PATH = "text_labels.pkl"
TEXT_EMBEDDINGS_PATH = "text_embeddings.idx"
IMAGE_STORAGE_PATH = "extracted_images.pkl"

# --- Load Embedding Model --- #
embedding_model =  SentenceTransformer('./models/all-MiniLM-L6-v2')

# --- Helper Functions --- #
def fetch_available_models():
    """Retrieve available local Ollama models."""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.ok:
            return [model.get("name") for model in response.json().get("models", [])]
    except requests.RequestException:
        pass
    return []

def extract_content_from_pdf(pdf_bytes, pdf_mode):
    """Extracts text and optionally images from a PDF."""
    text_accumulator = []
    extracted_images = []
    reader = PdfReader(io.BytesIO(pdf_bytes))

    for page in reader.pages:
        text = page.extract_text() or ""
        text_accumulator.append(text)

        if pdf_mode == "Standard Text PDF" and "/XObject" in page.get("/Resources", {}):
            xObjects = page["/Resources"]["/XObject"].get_object()
            for obj_name in xObjects:
                obj = xObjects[obj_name]
                if obj.get("/Subtype") == "/Image":
                    image_data = obj.get_data()
                    try:
                        img = Image.open(io.BytesIO(image_data))
                        extracted_images.append(img)
                        ocr_text = pytesseract.image_to_string(img)
                        text_accumulator.append(ocr_text)
                    except Exception:
                        continue  # Skip invalid images

    # Save extracted images
    with open(IMAGE_STORAGE_PATH, "wb") as img_file:
        pickle.dump(extracted_images, img_file)

    return "\n".join(text_accumulator).strip(), extracted_images

def build_vector_index(text_data):
    """Creates and saves FAISS index from text."""
    embeddings = embedding_model.encode([text_data])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, TEXT_EMBEDDINGS_PATH)
    with open(TEXT_LABELS_PATH, "wb") as label_file:
        pickle.dump(text_data, label_file)

    return index

def load_index_and_text():
    """Loads saved FAISS index and associated text."""
    if os.path.exists(TEXT_LABELS_PATH) and os.path.exists(TEXT_EMBEDDINGS_PATH):
        with open(TEXT_LABELS_PATH, "rb") as label_file:
            text_data = pickle.load(label_file)
        index = faiss.read_index(TEXT_EMBEDDINGS_PATH)
        return text_data, index
    return None, None

def load_saved_images():
    """Retrieves previously extracted images."""
    if os.path.exists(IMAGE_STORAGE_PATH):
        with open(IMAGE_STORAGE_PATH, "rb") as img_file:
            return pickle.load(img_file)
    return []

def query_llama_model(question, text_context, model_name):
    """Queries LLaMA-based local model using a prompt."""
    prompt = (
        f"Based on the following document, answer the question below in bullet points.\n"
        f"\nDocument Content:\n{text_context}\n\nQuestion:\n{question}"
    )
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=20,
        )
        if response.ok:
            return response.json().get("response", "No answer generated.")
    except requests.RequestException:
        pass
    return "Unable to connect to LLaMA server."

# --- Streamlit Application --- #
st.title("\U0001F4C4 AI-Powered Offline PDF Q&A System")

if st.button("Reset Session"):
    st.session_state.clear()
    st.rerun()

pdf_mode = st.selectbox("Choose PDF Processing Mode:", ["Standard Text PDF", "Handwritten Text PDF"])

available_models = fetch_available_models()
if available_models:
    selected_model = st.selectbox("Select AI Model:", available_models)
else:
    st.warning("No local models detected. Please start Ollama server.")
    selected_model = None

uploaded_pdf = st.file_uploader("Upload a PDF file:", type=["pdf"])

if uploaded_pdf and selected_model:
    # Save uploaded file temporarily for hashing
    pdf_bytes = uploaded_pdf.read()
    pdf_hash = str(hash(pdf_bytes))  # Simple hash to identify uniqueness

    idx_path = f"text_embeddings_{pdf_hash}.idx"
    lbl_path = f"text_labels_{pdf_hash}.pkl"
    img_path = f"extracted_images_{pdf_hash}.pkl"

    if os.path.exists(idx_path) and os.path.exists(lbl_path):
        st.info("Using cached data from disk.")
        with open(lbl_path, "rb") as f:
            pdf_text = pickle.load(f)
        with open(img_path, "rb") as f:
            pdf_images = pickle.load(f)
        index = faiss.read_index(idx_path)
    else:
        with st.spinner("Processing and indexing new PDF..."):
            pdf_text, pdf_images = extract_content_from_pdf(pdf_bytes, pdf_mode)
            embeddings = embedding_model.encode([pdf_text])
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))
            faiss.write_index(index, idx_path)
            with open(lbl_path, "wb") as f:
                pickle.dump(pdf_text, f)
            with open(img_path, "wb") as f:
                pickle.dump(pdf_images, f)
        st.success("New PDF processed and saved.")


    user_query = st.text_input("Enter your question about the document:")

    if st.button("Ask AI") and user_query:
        with st.spinner("Fetching AI Response..."):
            stored_text, _ = load_index_and_text()
            stored_images = load_saved_images()
            response_text = query_llama_model(user_query, stored_text, selected_model)

        st.subheader("\U0001F4AD AI Response:")
        st.info(response_text)

        if pdf_mode == "Standard Text PDF" and stored_images:
            st.subheader("\U0001F5BC Relevant Extracted Images:")
            for img in stored_images:
                st.image(img, caption="Extracted Image", width=400)
