# Multi-File RAG

Upload multiple documents (PDF, DOCX, TXT, MD, CSV) and ask questions across all of them.
Answers are generated with a Groq-hosted Llama model, grounded in the retrieved chunks, and
each answer lists which source file(s) it drew from.

Deployed on **Streamlit Community Cloud** (free) since Hugging Face Spaces now requires a
paid plan for Gradio/Docker SDKs.

## Deploy on Streamlit Community Cloud

1. Push this folder to a **GitHub repo** (public or private) containing:
   - `streamlit_app.py`
   - `requirements.txt`
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click
   **"New app"**.
3. Pick your repo, branch, and set the main file path to `streamlit_app.py`.
4. Before (or after) deploying, open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
5. Click **Deploy**. You'll get a public `*.streamlit.app` URL.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in your key
streamlit run streamlit_app.py
```

## How it works

- Files are loaded with the appropriate LangChain loader based on extension
  (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`, `UnstructuredMarkdownLoader`, `CSVLoader`).
- All documents are split into ~1000-character chunks and tagged with their source filename.
- Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in an
  in-memory FAISS index.
- On each question, the top 4 relevant chunks are retrieved and passed to
  `llama-3.1-8b-instant` (via Groq) along with the question, and the model answers using
  only that context.
