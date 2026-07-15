# 📚 Multi-File RAG Dashboard

A Streamlit app that combines a multi-format Retrieval-Augmented Generation (RAG) pipeline with a personal dashboard (to-do list, timetable) and **two-way Telegram integration**. Upload documents or audio, ask questions by typing, speaking, or texting a Telegram bot, and get answers grounded in your files, your app state, or both — with every reply citing its source.

---

## Table of contents

- [What it does](#what-it-does)
- [File overview](#file-overview)
- [Core RAG pipeline](#core-rag-pipeline)
- [Telegram integration](#telegram-integration)
- [Dashboard: timetable & to-dos](#dashboard-timetable--to-dos)
- [Optional Q&A logging to Google Sheets](#optional-qa-logging-to-google-sheets)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Using the app](#using-the-app)
- [Limitations & notes](#limitations--notes)
- [Troubleshooting](#troubleshooting)

---

## What it does

1. **Index documents/audio** — upload PDFs, Word docs, text, Markdown, CSVs, or audio files in the sidebar; the app transcribes audio (Groq Whisper) and builds a FAISS vector index over everything.
2. **Ask questions** — type a question, record one with your mic, or text your Telegram bot. Answers are grounded only in retrieved context (documents, dashboard state, or both) and always cite their source.
3. **Run a dashboard** — maintain a timetable and to-do list right in the app, and query them by voice, text, or Telegram (`/todos`, `/timetable`, `/summary`, ...).
4. **Feed content in via Telegram** — anything sent to your bots that *isn't* a recognized command is automatically embedded straight into the same RAG index, so forwarded articles, notes, or voice memos become searchable alongside your uploaded files.
5. **Optionally log every Q&A** to a Google Sheet for a running history.

## File overview

| File | Role |
|---|---|
| `streamlit_app.py` | The entire app: UI, RAG pipeline, Telegram polling/sending, dashboard widgets. |
| `telegram_assistant.py` | Pure logic for the main Telegram bot's command language (`/ask`, `/todos`, `/timetable`, `/important`, `/summary`, `/help`), dashboard-intent detection, and digest formatting. No Streamlit or network calls — easy to test in isolation. |
| `telegram_utils.py` *(imported, not reviewed here)* | Based on how it's called from `streamlit_app.py`, this appears to provide: `send_telegram_message(token, chat_id, text) -> (ok, info)`, `get_telegram_updates(token, offset=None) -> (updates, err)`, `extract_messages(updates) -> list[dict]` (each with `update_id`, `chat_id`, `from`, `text`, `date`), and `classify_message(text, keywords) -> "important" | "inbox"`. |
| `dashboard_utils.py` *(imported, not reviewed here)* | Based on its call sites, appears to provide simple CRUD helpers operating on `st.session_state`: `add_timetable_entry`, `remove_timetable_entry`, `add_todo`, `toggle_todo`, `remove_todo`. |

> The two "not reviewed" files weren't included in what I read — the descriptions above are inferred purely from how their functions are called in `streamlit_app.py`, not from their actual implementations. Worth double-checking against the real source if you're relying on exact behavior (e.g. what `classify_message` treats as a keyword match).

## Core RAG pipeline

### Loading & chunking
- `LOADER_MAP` routes `.pdf` → `PyPDFLoader`, `.docx` → `Docx2txtLoader`, `.txt` → `TextLoader`, `.md` → `UnstructuredMarkdownLoader`, `.csv` → `CSVLoader`.
- Audio extensions (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm`, `.mp4`, `.mpeg`, `.mpga`) go through `transcribe_audio()` instead, using Groq's `whisper-large-v3-turbo`.
- `load_any_file()` tags every resulting `Document` with `metadata["source"] = <original filename>`, which is what powers source citations later.
- **Silence filtering**: audio under `MIN_AUDIO_BYTES` (8000 bytes) is skipped outright as "almost certainly silent," and any transcript that exactly matches a known Whisper silence-hallucination phrase (`SILENCE_HALLUCINATIONS` — things like "Thank you.", "Thanks for watching!", "You", "Bye.") is discarded rather than indexed or treated as a real answer.
- `build_index()` collects all valid documents, splits them with `RecursiveCharacterTextSplitter` (1000-character chunks, 150-character overlap), embeds them with `sentence-transformers/all-MiniLM-L6-v2` (`HuggingFaceEmbeddings`, cached via `@st.cache_resource`), and stores them in a FAISS index. It returns the vectorstore, a status message, and a list of skipped files (with reasons — unsupported type, missing key, or silent audio).
- `add_text_to_index()` lets new text (e.g. an incoming Telegram message) be embedded and merged into an *existing* FAISS index on the fly, or create one if none exists yet — this is how Telegram content becomes searchable without rebuilding from scratch.

### Answering
Two answering paths exist:

- **`answer_question()`** — the original, documents-only path. Retrieves the top 4 chunks (`k=4`), formats them with `[Source: <file>]` headers, and runs them through a strict prompt (`PROMPT`) instructing the model to answer only from context, via `llama-3.1-8b-instant` (`ChatGroq`, `temperature=0`). Used by the in-app typed/voice question flow.
- **`answer_unified_query()`** — used for Telegram's `/ask`. Builds a **dashboard state snapshot** (todos, timetable, recent important messages) via `telegram_assistant.build_state_context()`, retrieves document context the same way as above, and feeds *both* into `COMBINED_PROMPT`. That prompt enforces explicit priority rules:
  1. Todo/schedule/"important messages" questions are answered **only** from dashboard state, never documents.
  2. If the relevant dashboard section is empty, say so plainly — don't fall back to guessing from documents.
  3. Everything else is answered from document context, with the source file cited.
  4. If neither source has the answer, say so rather than guessing.

Before even reaching the LLM, `telegram_assistant.detect_dashboard_intent()` tries to shortcut obviously dashboard-shaped questions (containing words like "todo", "timetable", "important", "summary") straight to a deterministic formatted answer — *unless* the question also references documents/files (e.g. "is there anything important in the contract"), in which case it always falls through to the full RAG pipeline.

## Telegram integration

There are **two independent bots**, each with a distinct job:

### Main bot (Q&A + commands)
Configured with a **Bot Token** and **Chat ID** in the "3. Telegram" section. Polling (`poll_main_bot_once()`) fetches new updates and, for each message:
- If it's a recognized command, it's answered **directly back on Telegram**:
  - `/ask <question>` or `?<question>` — routes through `answer_unified_query()` (documents + dashboard state)
  - `/todos`, `/timetable`, `/important`, `/summary`, `/help` — answered deterministically from `telegram_assistant`'s formatters
- Otherwise, the message text is embedded straight into the shared RAG index via `add_text_to_index()`, tagged with source `Telegram (<sender>)`.
- Replies over 4000 characters are truncated (Telegram's hard cap is 4096).
- Every `/ask`-style Q&A is also recorded into `st.session_state.telegram_commands` and logged to Google Sheets (if configured).
- Can be checked manually ("Check for new messages now") or on an automatic timer (`st.fragment(run_every=...)`, configurable 10–300s) as long as the tab stays open.

### Content bot (pure intake, optional)
A second bot, meant for forwarding notes/articles/voice clips without expecting a reply. `poll_content_bot_once()`:
- Classifies each incoming message as **Inbox** or **Important** using a comma-separated keyword list you configure in the UI (default: `cricket, sports, research`).
- Embeds every message into the same RAG index (`Telegram Content Bot (<sender>)` as the source), regardless of which bucket it landed in.
- Same manual/auto-poll pattern as the main bot (10–600s interval).

### Command reference (from `telegram_assistant.py`)

| Command | Behavior |
|---|---|
| `/ask <question>` or `?<question>` | Full RAG + dashboard-state answer |
| `/todos` | Current to-do list |
| `/timetable` | Today's schedule |
| `/important` | Last 10 important messages |
| `/summary` | Condensed digest of todos, timetable, and important messages |
| `/help` | Lists all commands |

**A note on "always-on"**: both bots only poll while the Streamlit app is open in a browser tab (and awake — e.g. Streamlit Community Cloud's free tier sleeps idle apps). This is *not* a 24/7 background service; leave a tab open for polling to keep working.

## Dashboard: timetable & to-dos

Rendered in the sidebar-adjacent column:
- **Timetable**: add an entry with a time + activity via a form; each entry can be removed individually. Entries are plain `{time, activity, id}` dicts managed by `dashboard_utils`.
- **To-Do List**: add tasks via a form; each has a checkbox to toggle done/not-done and a remove button. A completion count (`x/y completed`) is shown once at least one task is done.

Both are queryable from the in-app UI implicitly (via the unified RAG answer) and explicitly via Telegram (`/todos`, `/timetable`, `/summary`).

## Optional Q&A logging to Google Sheets

- `get_log_worksheet()` looks for `gcp_service_account` and `GSHEET_URL` in Streamlit secrets. If either is missing, logging is simply treated as "not configured" — not an error.
- When present, it authorizes a Google service account (Sheets + Drive scopes), opens the sheet by URL, and adds a `timestamp, question, answer` header row if the sheet is empty.
- `log_qa()` appends a row after every answered question — both from the in-app UI and from Telegram's `/ask`. Any failure here is caught and surfaced only as a small caption; **logging never breaks the main app.**

## Requirements

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys) — for the LLM and for transcribing any audio (uploads or voice questions)
- One or two [Telegram bot tokens](https://core.telegram.org/bots#botfather) (via @BotFather) if you want Telegram integration
- (Optional) A Google Cloud service account for Q&A logging to Sheets

Based on the imports actually present in the code, the app depends on at least:

```
streamlit
langchain-core
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-groq
groq
faiss-cpu
sentence-transformers
gspread
google-auth
```

plus the format-specific packages the document loaders need under the hood (typically `pypdf`, `docx2txt`, `unstructured`/`markdown`), and whatever `telegram_utils.py` uses to talk to the Telegram Bot API (likely `requests` or similar — not confirmed, since that file wasn't reviewed here). Install from the repo's own `requirements.txt` for exact, pinned versions.

## Configuration

Read from Streamlit secrets first, falling back to environment variables (for a couple of keys), and finally to whatever's typed into the UI at runtime.

| Key | Required? | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes (or typed into the sidebar) | LLM answers + audio transcription |
| `GSHEET_URL` | No | Sheet URL for Q&A logging |
| `gcp_service_account` | No | Service-account JSON (as a TOML table) for Sheets/Drive |
| `TELEGRAM_BOT_TOKEN` | No (or typed into the UI) | Main bot's token |
| `TELEGRAM_CHAT_ID` | No (or typed into the UI) | Default chat ID the main bot replies to |
| `TELEGRAM_CONTENT_BOT_TOKEN` | No (or typed into the UI) | Content bot's token |

Example `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your-groq-api-key-here"
TELEGRAM_BOT_TOKEN = "your-main-bot-token"
TELEGRAM_CHAT_ID = "your-chat-id"
TELEGRAM_CONTENT_BOT_TOKEN = "your-content-bot-token"

GSHEET_URL = "https://docs.google.com/spreadsheets/d/your-sheet-id"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "..."
client_email = "..."
client_id = "..."
# ...remaining fields from the downloaded service-account JSON
```

## Setup

```bash
git clone https://github.com/waqas-manzoor5595/multi-file-rag.git
cd multi-file-rag
pip install -r requirements.txt
```

To get Telegram bot tokens: message [@BotFather](https://t.me/BotFather) on Telegram, run `/newbot` twice (once for the main bot, optionally again for the content bot), and copy each token it gives you. Your Chat ID can be found by messaging your bot once and checking the `chat.id` field in a manual `getUpdates` call, or via a helper bot like @userinfobot.

## Running the app

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501` by default.

## Using the app

1. **Build the index**: upload files in the sidebar, then click **Build Index**.
2. **Ask in-app**: type a question or record one with the mic under "2. Ask a question," then click **Ask**. The answer and its expandable source chunks appear below.
3. **Use the dashboard**: add timetable entries and to-dos in the right-hand column; check them off or remove them as needed.
4. **Connect Telegram**: enter your bot token(s) and chat ID under "3. Telegram," optionally enable auto-polling, and either click **Check for new messages now** or let it poll on a timer.
5. **Text your bot**: send `/help` to see available commands, `/ask <question>` (or just `?<question>`) for anything else, or forward content to your content bot to have it silently indexed and sorted into Inbox/Important.

## Limitations & notes

- The FAISS index is **in-memory and per-session** — nothing persists across app restarts.
- Retrieval always pulls the top 4 chunks (`k=4`); not configurable from the UI.
- Chunking is fixed at 1000 characters with 150-character overlap.
- Telegram polling (both bots) only happens while the app tab is open and awake — there's no persistent background worker.
- Dashboard-intent detection (`detect_dashboard_intent`) is a heuristic keyword match, not a classifier — it explicitly defers to the full RAG path whenever a question also mentions documents/files, but edge cases in phrasing could still go either way.
- Silence-hallucination filtering for audio is a fixed list of known Whisper filler phrases plus a minimum byte-size cutoff; unusual but genuinely silent clips could in theory still slip through if Whisper hallucinates something not on that list.
- Logging failures, Telegram send/fetch failures, and per-message ingestion failures are all caught and reported without crashing the app.

## Troubleshooting

- **"No GROQ API key found"** — set `GROQ_API_KEY` in secrets/env, or paste it into the sidebar.
- **Audio/voice question skipped or warned as "very short or silent"** — the recording was under the 8000-byte threshold; check mic permissions and speak for a couple of seconds before stopping.
- **"Couldn't extract text from any uploaded file"** — every file was unsupported, silent (audio), or otherwise produced no text; check the skipped-files warning.
- **`/ask` on Telegram replies "no GROQ API key is configured"** — the key field in the sidebar (`groq_api_key` session state) was empty at poll time.
- **Telegram commands aren't being answered** — confirm the bot token and chat ID are correct, and that either manual "Check for new messages now" or auto-polling (tab open) is actually running.
- **Logging shows "off — connection failed"** — verify the service account was shared as an Editor on the sheet, and that both the Sheets and Drive APIs are enabled in Google Cloud.
