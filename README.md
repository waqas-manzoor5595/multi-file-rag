# Multi-File RAG

Upload multiple documents (PDF, DOCX, TXT, MD, CSV) **or audio files** and ask questions
across all of them — by typing or by speaking. Audio (files or recorded questions) is
transcribed using Groq's Whisper model. Answers are generated with a Groq-hosted Llama
model, grounded in the retrieved chunks, and each answer lists which source file(s) it
drew from.

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

## Set up Q&A logging to Google Sheets (optional)

Every question and answer can be logged to a Google Sheet automatically. This is optional —
without it configured, the app works exactly the same, just without logging.

### 1. Create the Google Sheet
1. Go to [sheets.google.com](https://sheets.google.com) and create a new blank sheet.
2. Name it whatever you like, e.g. "RAG Q&A Log".
3. Copy its URL from the address bar — you'll need it below.

### 2. Create a Google Cloud service account
1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new
   project (or use an existing one).
2. In the search bar, find and enable these two APIs:
   - **Google Sheets API**
   - **Google Drive API**
3. Go to **APIs & Services → Credentials → Create Credentials → Service account**.
4. Give it any name (e.g. "rag-sheet-logger") and click through to **Done**.
5. Click on the service account you just created → **Keys** tab → **Add Key → Create new key**
   → choose **JSON** → **Create**. This downloads a `.json` file — keep it safe, it's a
   credential.

### 3. Share the Sheet with the service account
1. Open the downloaded JSON file and copy the `client_email` value
   (looks like `something@your-project.iam.gserviceaccount.com`).
2. Open your Google Sheet → **Share** → paste that email in → give it **Editor** access → Send.

### 4. Add the credentials to Streamlit secrets
1. Open your app on Streamlit Cloud → **⋮ menu → Settings → Secrets**.
2. Add your `GROQ_API_KEY` (if not already there), the Sheet URL, and the full contents of
   the JSON key file under `[gcp_service_account]`. See `.streamlit/secrets.toml.example`
   in this repo for the exact format to paste in — copy each field from the downloaded JSON
   into the matching line.
3. Save. The app restarts automatically and the sidebar will show "Logging: questions &
   answers are being saved to Google Sheets."

Each row logged is `timestamp, question, answer` — no files or personal info beyond what's
typed or spoken as a question.

## How it works

- Files are loaded with the appropriate LangChain loader based on extension
  (`PyPDFLoader`, `Docx2txtLoader`, `TextLoader`, `UnstructuredMarkdownLoader`, `CSVLoader`).
- **Audio files** (mp3, wav, m4a, ogg, flac, webm, mp4) are transcribed with Groq's
  `whisper-large-v3-turbo` model and the transcript is indexed like any other document.
- **Voice questions**: click the microphone in the question section to record and ask
  a question out loud — it's transcribed the same way before being sent to the RAG chain.
- If Google Sheets logging is configured, every question and its answer are appended as a
  new row with a timestamp. Logging failures never break the app — they're silently skipped.
- All documents are split into ~1000-character chunks and tagged with their source filename.
- Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in an
  in-memory FAISS index.
- On each question, the top 4 relevant chunks are retrieved and passed to
  `llama-3.1-8b-instant` (via Groq) along with the question, and the model answers using
  only that context.

