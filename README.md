# 🎙 VoiceRAG — Voice-Enabled RAG with Sarvam AI + Groq

A full-stack Voice-Enabled Retrieval-Augmented Generation (RAG) app.  
**STT:** Sarvam AI (`saaras:v3`) | **LLM:** Groq (`llama3-8b-8192`) | **Retrieval:** FAISS

---

## ✅ Quick Start

### 1. Clone & set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
SARVAM_API_KEY="your_sarvam_api_key_here"   # https://dashboard.sarvam.ai/
GROQ_API_KEY="your_groq_api_key_here"        # https://console.groq.com/
```

---

### 2. Install Python dependencies

> Requires Python 3.10+

```bash
pip install -r requirements.txt
```

---

### 3. Build the FAISS knowledge index

```bash
python build_index.py --lang hi --max-docs 100
```

This creates `msmarco_faiss.index` and `msmarco_chunks.csv` in the project root.  
*(Already included in the repo — only re-run if you want to rebuild or change the language.)*

---

### 4. Start the FastAPI backend

```bash
uvicorn server:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
API docs: `http://localhost:8000/docs`

---

### 5. Install & start the React frontend

```bash
cd frontend
npm install
npm run dev
```

The app opens at **http://localhost:5173**.

---

## 🏗️ Project Structure

```
.
├── server.py           # FastAPI backend (main entry point)
├── stt.py              # Sarvam AI Speech-to-Text client
├── llm.py              # Groq LLM client (llama3-8b-8192)
├── retriever.py        # FAISS-based semantic retriever
├── app.py              # CLI pipeline (benchmark / single test)
├── build_index.py      # Script to build the FAISS index
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this!)
├── .env.example        # Template for .env
├── msmarco_faiss.index # Pre-built FAISS index
├── msmarco_chunks.csv  # Chunked document metadata
└── frontend/           # React + Vite frontend
    ├── vite.config.js  # Vite config (proxy + React plugin)
    ├── package.json
    └── src/
        ├── App.jsx
        ├── App.css
        ├── main.jsx
        └── components/
            ├── Header.jsx
            ├── QueryInput.jsx   # Real mic recording → Sarvam STT
            ├── AnswerCard.jsx
            ├── MetricsPanel.jsx
            └── History.jsx
```

---

## 🎤 Voice Input

The mic button uses **real microphone recording** via the browser's `MediaRecorder` API.  
Recorded audio is sent to `/api/voice` → transcribed by **Sarvam AI** → fed into the RAG pipeline.

- Tap once to **start** recording
- Tap again (or wait 8 seconds) to **stop** and process

> **Requires `SARVAM_API_KEY`** for live voice. Without a key the app falls back to mock mode.

---

## 🔬 Mock Mode vs Live Mode

Toggle the **Mock/Live** button in the top-right corner.

| Mode | STT | LLM |
|------|-----|-----|
| Mock | Returns a fixed test query | Returns a canned answer |
| Live | Sarvam AI (`saaras:v3`) | Groq (`llama3-8b-8192`) |

FAISS retrieval is always real (local, no API key needed).

---

## 🧪 Benchmarking (CLI)

```bash
# Run 20-query latency benchmark (mock mode)
python app.py --benchmark --mock

# Single live query test
python app.py
```

---

## 🚀 Production Deployment

1. **Build the frontend:**
   ```bash
   cd frontend && npm run build
   ```
2. **Serve static files from FastAPI** (optional, add `StaticFiles` mount) or deploy `frontend/dist/` to a CDN.
3. **Run the backend with a production server:**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2
   ```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/query` | Text query → RAG pipeline |
| `POST` | `/api/voice` | Audio upload → Sarvam STT → RAG pipeline |
| `GET` | `/docs` | Interactive Swagger UI |
