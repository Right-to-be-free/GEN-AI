# 🧠 Local File + Image Semantic Search Engine with Vector Database

**Author:** Rishi Vishal  
**Last Updated:** June 12, 2025  
**Tech Stack:** Python, SentenceTransformers, Pinecone, Tesseract OCR, Pytesseract, Pillow

---

## 🚀 Overview

This project is my personal journey into building a **production-grade semantic vector search engine**. I designed and implemented a system that can:

- 🗂️ Ingest and process multiple file types (`.txt`, `.pdf`, `.csv`, `.docx`, `.xlsx`)
- 🖼️ Extract text from images (`.png`, `.jpg`, `.jpeg`) using Tesseract OCR
- ✂️ Chunk long texts for better semantic embedding
- 🧠 Generate embeddings using `all-MiniLM-L6-v2`
- 🌲 Store vector representations in Pinecone
- 🔍 Query the vector DB using natural language
- 🧼 Deduplicate content using text hashing
- 🧪 Test everything from the terminal (CLI)

---

## 🛠️ Setup Instructions

### 🔧 Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 🧪 Required Installs

```bash
pip install sentence-transformers pinecone python-dotenv pytesseract pillow
```

Also install **Tesseract OCR for Windows**:  
👉 https://github.com/UB-Mannheim/tesseract/wiki  
Set the path in `file_loader.py`:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## 📁 Folder Structure

```
vector-db-project/
├── Files/                  # All input files
├── chunker.py              # Text chunking logic
├── deduplicator.py         # Tracks and hashes processed content
├── embedding_generator.py  # MiniLM embedding model
├── file_loader.py          # File type loader + OCR
├── main.py                 # CLI-based orchestrator
├── pinecone_handler.py     # Pinecone init, upsert, and query
├── processed_files.json    # Stores hashes of processed files
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔄 Supported File Types

| Type        | Method             |
|-------------|--------------------|
| `.txt`      | Plain text reader  |
| `.pdf`      | PyMuPDF            |
| `.csv`      | pandas             |
| `.docx`     | python-docx        |
| `.xlsx`     | pandas + openpyxl  |
| `.jpg/.png` | Tesseract OCR      |

---

## ✅ Features I Implemented

### 📄 File Ingestion Pipeline
- Load files from `Files/` or via terminal path
- Extract, chunk, embed, and upsert

### 🔍 Semantic Search
- Natural language queries
- Results retrieved from Pinecone with metadata

### 🧠 Deduplication
- Store and compare file content hash
- Only updates modified or new files

### 🖼️ Image OCR
- Integrated Tesseract OCR
- Converted image text to embeddings

### 🧪 CLI Modes
- `ingest`: batch ingest from Files folder
- `single`: ingest just one file
- `query`: semantic search from Pinecone

---

## 📤 How I Use It

```bash
python main.py
# Choose: ingest | query | single
```

- `single`: Drop any file path (text/image/doc)
- `query`: Type in natural language
- All results log to console and are visible in Pinecone UI

---

## 🏆 What I Achieved

- ✅ Built a custom vector DB for mixed-format documents
- ✅ Enabled semantic search over my personal files
- ✅ Learned how to use OCR, chunking, and embeddings
- ✅ Connected to Pinecone with real-time upsert/query
- ✅ Created a maintainable, extensible backend

---

## 🔮 Next Steps (Planned)

- 🌐 Streamlit interface
- 💬 Chatbot mode using GPT + Pinecone context
- 🧼 Auto-clean OCR noise
- 🧪 Deployable REST API version

---

## 📬 Contact

If you're learning or building something similar, feel free to connect!

---

