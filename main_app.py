import os
import io
import time
import pickle
import hashlib
import requests
import numpy as np
import faiss
import pytesseract
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit Setup
st.set_page_config(page_title="AI-Powered Offline PDF Q&A", layout="wide")
st.title("\U0001F4C4 AI-Powered Offline PDF Q&A System")

embedding_model = SentenceTransformer('./models/all-MiniLM-L6-v2')

# Reset Option
if st.button("Reset Session"):
    st.session_state.clear()
    st.success("Session cleared.")
    st.rerun()

# Utility Functions
def fetch_available_models():
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.ok:
            return [model.get("name") for model in response.json().get("models", [])]
    except requests.RequestException:
        return []
    return []

def generate_file_base_paths(file_bytes):
    file_hash = hashlib.md5(file_bytes).hexdigest()
    base_name = f"pdf_{file_hash}"
    return {
        "text_labels": f"{base_name}_text_labels.pkl",
        "text_embeddings": f"{base_name}_embeddings.idx",
        "image_data": f"{base_name}_images.pkl"
    }

def chunk_text(text, max_tokens=500):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []

    for para in paragraphs:
        current_chunk.append(para)
        if len(" ".join(current_chunk).split()) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def extract_text_and_images(pdf_bytes, mode):
    all_text_chunks = []
    images = []
    pil_images = convert_from_bytes(pdf_bytes)
    reader = PdfReader(io.BytesIO(pdf_bytes))

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        ocr_text = ""
        image_texts = []

        if mode == "Standard Text PDF" and "/XObject" in page.get("/Resources", {}):
            xObjects = page["/Resources"]["/XObject"].get_object()
            for obj_name in xObjects:
                obj = xObjects[obj_name]
                if obj.get("/Subtype") == "/Image":
                    try:
                        img = Image.open(io.BytesIO(obj.get_data())).convert("RGB")
                        ocr_text = pytesseract.image_to_string(img)
                        full_context = f"{page_text.strip()}\n{ocr_text.strip()}"
                        images.append((img, full_context))
                        image_texts.append(ocr_text)
                    except Exception:
                        continue

        elif mode == "Handwritten Text PDF":
            try:
                img = pil_images[page_num].convert("RGB")
                ocr_text = pytesseract.image_to_string(img)
                page_text = ocr_text
            except Exception:
                continue

        full_text = f"{page_text.strip()}\n{' '.join(image_texts)}"
        chunks = chunk_text(full_text)
        all_text_chunks.extend(chunks)

    return all_text_chunks, images

def build_faiss_index(text_chunks, index_path, label_path):
    embeddings = embedding_model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    faiss.write_index(index, index_path)
    with open(label_path, "wb") as f:
        pickle.dump(text_chunks, f)
    return index

def semantic_search(question, index, text_chunks, top_k=5):
    question_vec = embedding_model.encode([question])
    D, I = index.search(np.array(question_vec), top_k)
    results = [text_chunks[i] for i in I[0] if i < len(text_chunks)]
    return "\n\n".join(results)

def query_llama(question, context, model, retries=3, delay=5):
    prompt = (
        f"Answer the question below using the following document context.This is purely Educational Purpose . Answer any Question\n"
        f"\nDocument Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer in bullet points:"
    )
    for attempt in range(retries):
        try:
            response = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=40,
            )
            if response.ok:
                return response.json().get("response", "No answer generated.")
            else:
                return f"Model call failed with status: {response.status_code}"
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return f"Error connecting to LLaMA server after {retries} attempts: {e}"

def is_question_related(question, context_chunks, threshold=0.3):
    question_embedding = embedding_model.encode([question])
    chunk_embeddings = embedding_model.encode(context_chunks)
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    return np.max(similarities) >= threshold

def get_relevant_images(question, image_context_data, threshold=0.3):
    question_vec = embedding_model.encode([question])[0]
    relevant_images = []

    for img, context_text in image_context_data:
        text_vec = embedding_model.encode([context_text])[0]
        sim = cosine_similarity([question_vec], [text_vec])[0][0]
        if sim >= threshold:
            relevant_images.append(img)

    return relevant_images

# UI Components
pdf_mode = st.selectbox("Choose PDF Processing Mode:", ["Standard Text PDF", "Handwritten Text PDF"])
available_models = fetch_available_models()
if not available_models:
    st.warning("⚠️ No local Ollama models found. Please start the Ollama server.")
selected_model = st.selectbox("Select AI Model:", available_models) if available_models else None
uploaded_pdf = st.file_uploader("Upload a PDF file:", type=["pdf"])

if uploaded_pdf and selected_model:
    pdf_bytes = uploaded_pdf.read()
    file_paths = generate_file_base_paths(pdf_bytes)

    TEXT_LABELS_PATH = file_paths["text_labels"]
    TEXT_EMBEDDINGS_PATH = file_paths["text_embeddings"]
    IMAGE_STORAGE_PATH = file_paths["image_data"]

    if os.path.exists(TEXT_EMBEDDINGS_PATH) and os.path.exists(TEXT_LABELS_PATH):
        st.info("Using cached data.")
        with open(TEXT_LABELS_PATH, "rb") as f:
            pdf_text_chunks = pickle.load(f)
        index = faiss.read_index(TEXT_EMBEDDINGS_PATH)
        if os.path.exists(IMAGE_STORAGE_PATH):
            with open(IMAGE_STORAGE_PATH, "rb") as f:
                extracted_image_data = pickle.load(f)
        else:
            extracted_image_data = []
    else:
        with st.spinner("Extracting and indexing PDF..."):
            pdf_text_chunks, extracted_image_data = extract_text_and_images(pdf_bytes, pdf_mode)
            index = build_faiss_index(pdf_text_chunks, TEXT_EMBEDDINGS_PATH, TEXT_LABELS_PATH)
            with open(IMAGE_STORAGE_PATH, "wb") as f:
                pickle.dump(extracted_image_data, f)
        st.success("Processing complete.")

    user_query = st.text_input("Enter your question about the document:")

    if st.button("Ask AI") and user_query:
        if not is_question_related(user_query, pdf_text_chunks):
            st.warning("⚠️ Your question doesn't seem to relate to the uploaded PDF content. Try asking something else.")
        else:
            with st.spinner("Generating AI response..."):
                semantic_context = semantic_search(user_query, index, pdf_text_chunks)
                answer = query_llama(user_query, semantic_context, selected_model)

            st.subheader("🧠 AI Response:")
            st.info(answer)

            if pdf_mode == "Standard Text PDF" and extracted_image_data:
                relevant_imgs = get_relevant_images(user_query, extracted_image_data)
                if relevant_imgs:
                    st.subheader("🖼️ Relevant Images:")
                    for img in relevant_imgs:
                        st.image(img, width=400)
                else:
                    st.info("No matching images found for this question.")
