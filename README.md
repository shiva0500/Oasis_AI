

# 🌴 Oasis AI: Offline Research Companion

> Your fully offline PDF-based Question & Answer tool powered by LLaMA and local embeddings.

---

## 📖 Overview

**Oasis AI** is an offline research assistant designed for students, researchers, and professionals who work with PDFs. It allows you to upload any PDF document and ask questions about its contents — completely offline. No internet required, and your data never leaves your machine.

---

## 🧠 Features

- ✅ PDF text and image extraction (supports scanned/handwritten PDFs)
- ✅ Embedding generation with SentenceTransformers
- ✅ Local FAISS vector index for semantic search
- ✅ AI question-answering via local LLaMA models (Ollama integration)
- ✅ Auto-caching and fast reloading
- ✅ Built with Streamlit for a user-friendly experience

---

## 💾 Requirements

- Python 3.9+
- Dependencies (see below)
- Local Ollama server running with a LLaMA-based model installed (e.g., llama3, llama2, mistral)

---

## ⚙️ Installation

### Step 1: Clone this repo or download the installer

```bash
git clone https://github.com/yourusername/oasis-ai
cd oasis-ai
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

Required libraries:

- streamlit
- sentence-transformers
- faiss-cpu
- PyPDF2
- pdf2image
- pytesseract
- Pillow
- requests

You also need:

- Tesseract OCR installed and in your system PATH
- Ollama (https://ollama.com)

### Step 3: Run the app

```bash
streamlit run launch_app.py
```

---

## 🗂 File Structure

```
oasis-ai/
│
├── launch_app.py                # Main Streamlit app
├── models/                      # Embedding models
├── text_labels.pkl              # Saved extracted text
├── text_embeddings.idx          # FAISS index file
├── extracted_images.pkl         # Images from PDF (if any)
├── README.md                    # This file
└── dist/                        # (Optional) Packaged executable
```

---

## 🚀 How to Use

1. Start your Ollama server:
   ```bash
   ollama run llama3
   ```

2. Open Oasis AI and:
   - Upload a PDF
   - Select your AI model
   - Ask a question about the content!

3. It will extract text (and images if available), create embeddings, and answer based on your PDF.

---

## 🔒 Privacy First

All processing is done **entirely offline**. Your documents are never uploaded or tracked.

---

## 🛠 Troubleshooting

- ❗ No local models detected? → Ensure Ollama is running: `http://127.0.0.1:11434`
- 🧠 Not getting smart answers? → Try using higher-quality models like `llama3`
- 🧹 App crashing? → Delete `.pkl` and `.idx` files to reset cache

---

## 📦 Build Executable (Optional)

Use PyInstaller to create a standalone `.exe`:

```bash
pyinstaller --noconfirm --onefile --name "launch_app" launch_app.py
```

Then package with the included Inno Setup script to distribute easily.

---

## 📄 License

This project is open-source and for educational/research use only. You are responsible for the data you process and local model usage.

---
