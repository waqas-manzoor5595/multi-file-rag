import os
import tempfile
from datetime import datetime

import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from groq import Groq

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from telegram_utils import (
    send_telegram_message,
    get_telegram_updates,
    extract_messages,
    classify_message,
)
from dashboard_utils import (
    add_timetable_entry,
    remove_timetable_entry,
    add_todo,
    toggle_todo,
    remove_todo,
)
import telegram_assistant as tga

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".mpeg", ".mpga"}
TRANSCRIBE_MODEL = "whisper-large-v3-turbo"

PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. Each context chunk is labeled with its source file.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

COMBINED_PROMPT = ChatPromptTemplate.from_template(
    """You are a personal assistant with access to the user's uploaded documents AND
their current dashboard state (to-do list, timetable, and flagged important Telegram messages).

Use whichever is relevant to answer the question. If it's about documents, cite the source file
by name. If it's about todos, timetable, or messages, answer directly from the dashboard state
below. If the answer isn't in either, say so plainly rather than guessing.

Document context:
{doc_context}

Dashboard state:
{state_context}

Question: {question}

Answer (keep it concise — this will be sent back as a Telegram message):"""
)


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_log_worksheet():
    """Connect to the Google Sheet used for logging Q&A.

    Returns (worksheet, error) — worksheet is None if logging is unavailable,
    and error explains why (or is None if simply not configured)."""
    try:
        service_account_info = st.secrets["gcp_service_account"]
        sheet_url = st.secrets["GSHEET_URL"]
    except Exception:
        return None, None  # not configured — not an error, just off

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    try:
        creds = Credentials.from_service_account_info(dict(service_account_info), scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url)
        worksheet = sheet.sheet1
        # Add a header row if the sheet is empty
        if not worksheet.get_all_values():
            worksheet.append_row(["timestamp", "question", "answer"])
    except Exception as e:
        return None, str(e)
    return worksheet, None


def log_qa(question, answer):
    """Best-effort logging: never let a logging failure break the app."""
    worksheet, _ = get_log_worksheet()
    if worksheet is None:
        return
    try:
        worksheet.append_row([datetime.utcnow().isoformat(), question, answer])
    except Exception as e:
        st.caption(f"(Couldn't log this Q&A to Google Sheets: {e})")


def transcribe_audio(path, api_key):
    """Send an audio file to Groq's Whisper endpoint and return the transcript text."""
    client = Groq(api_key=api_key)
    with open(path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            file=(os.path.basename(path), f.read()),
            model=TRANSCRIBE_MODEL,
        )
    return transcript.text


SILENCE_HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching!",
    "thanks for watching.", "you", "bye.",
}
MIN_AUDIO_BYTES = 8000  # below this, a recording is almost certainly silent/empty


def load_any_file(path, display_name, api_key=None):
    ext = os.path.splitext(display_name)[1].lower()

    if ext in AUDIO_EXTENSIONS:
        if not api_key:
            return []
        if os.path.getsize(path) < MIN_AUDIO_BYTES:
            # Too small to contain real speech — skip rather than let Whisper
            # hallucinate a generic phrase like "Thank you." on silence.
            return []
        text = transcribe_audio(path, api_key)
        if not text or not text.strip():
            return []
        if text.strip().lower() in SILENCE_HALLUCINATIONS:
            # Almost certainly a silence hallucination, not real content.
            return []
        return [Document(page_content=text, metadata={"source": display_name})]

    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        return []
    docs = loader_cls(path).load()
    for d in docs:
        d.metadata["source"] = display_name
    return docs


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


def add_text_to_index(vectorstore, text, source_label):
    """Embed a piece of text (e.g. a Telegram message) and add it to an
    existing FAISS vectorstore, or create a new one if none exists yet.
    Returns the (possibly newly created) vectorstore."""
    if not text or not text.strip():
        return vectorstore
    doc = Document(page_content=text, metadata={"source": source_label})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents([doc])
    if vectorstore is None:
        return FAISS.from_documents(chunks, get_embeddings())
    vectorstore.add_documents(chunks)
    return vectorstore


def build_index(uploaded_files, api_key):
    all_docs = []
    skipped = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uf in uploaded_files:
            ext = os.path.splitext(uf.name)[1].lower()
            if ext not in LOADER_MAP and ext not in AUDIO_EXTENSIONS:
                skipped.append(uf.name)
                continue
            if ext in AUDIO_EXTENSIONS and not api_key:
                skipped.append(f"{uf.name} (needs GROQ key to transcribe)")
                continue

            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())

            docs = load_any_file(tmp_path, uf.name, api_key)
            if ext in AUDIO_EXTENSIONS and not docs:
                skipped.append(f"{uf.name} (silent/no speech detected)")
                continue
            all_docs.extend(docs)

    if not all_docs:
        return None, "Couldn't extract text from any uploaded file.", skipped

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    indexed_count = len(uploaded_files) - len(skipped)
    msg = f"Indexed {len(chunks)} chunks from {indexed_count} file(s)."
    return vectorstore, msg, skipped


def poll_main_bot_once(bot_token, default_chat_id, api_key):
    """Fetch new messages for the main bot: answer commands (/ask, /todos, etc.)
    directly on Telegram, and embed everything else straight into the RAG
    index (no Inbox/Important sorting — that's the Content Bot's job).
    Returns (answered, ingested_count, failures, error). Never raises."""
    if st.session_state.get("_main_poll_in_progress"):
        return 0, 0, [], None
    st.session_state["_main_poll_in_progress"] = True
    try:
        offset = (
            st.session_state.telegram_last_update_id + 1
            if st.session_state.telegram_last_update_id is not None
            else None
        )
        updates, err = get_telegram_updates(bot_token, offset=offset)
        if err:
            return 0, 0, [], err

        messages = extract_messages(updates)
        answered, ingested_count, failures = 0, 0, []

        for m in messages:
            st.session_state.telegram_last_update_id = max(
                st.session_state.telegram_last_update_id or 0, m["update_id"]
            )
            reply_chat = m["chat_id"] or default_chat_id

            try:
                if tga.is_command(m["text"]):
                    cmd, arg = tga.parse_command(m["text"])
                    if cmd == "todos":
                        reply = "✅ To-Do List\n\n" + tga.format_todos(st.session_state.todos)
                    elif cmd == "timetable":
                        reply = "🗓️ Timetable\n\n" + tga.format_timetable(st.session_state.timetable)
                    elif cmd == "important":
                        reply = "⭐ Important Messages\n\n" + tga.format_important(
                            st.session_state.telegram_important, limit=10
                        )
                    elif cmd == "summary":
                        reply = tga.build_digest(st.session_state)
                    elif cmd == "help":
                        reply = tga.HELP_TEXT
                    else:  # "ask"
                        if not api_key.strip():
                            reply = "I can't answer that — no GROQ API key is configured."
                        elif not arg.strip():
                            reply = "Ask me something after /ask — e.g. `/ask what's on my schedule today?`"
                        else:
                            state_ctx = tga.build_state_context(st.session_state)
                            answer, _ = answer_unified_query(
                                st.session_state.vectorstore, arg, api_key.strip(), state_ctx
                            )
                            reply = answer or "(No answer was generated — try rephrasing the question.)"
                            st.session_state.telegram_commands.append(
                                {"question": arg, "answer": reply, "date": m.get("date")}
                            )

                    if len(reply) > 4000:  # Telegram caps messages at 4096 characters
                        reply = reply[:4000] + "\n\n[...truncated]"

                    ok, info = send_telegram_message(bot_token, str(reply_chat), reply)
                    if ok:
                        answered += 1
                    else:
                        failures.append(f"reply to \"{m['text'][:40]}\": {info}")
                else:
                    st.session_state.vectorstore = add_text_to_index(
                        st.session_state.vectorstore, m["text"], source_label=f"Telegram ({m['from']})"
                    )
                    ingested_count += 1
            except Exception as e:
                failures.append(f"\"{m['text'][:40]}\": {e}")

        return answered, ingested_count, failures, None
    except Exception as e:
        return 0, 0, [], str(e)
    finally:
        st.session_state["_main_poll_in_progress"] = False


def poll_content_bot_once(token, important_keywords):
    """Fetch any new messages sent to the content bot, sort each into
    Inbox/Important based on keywords, and embed them into the RAG index.
    Returns (ingested_count, error) — error is None on success.
    Never raises: any failure is returned as an error string instead, so a
    background poll can't crash the app."""
    if st.session_state.get("_content_poll_in_progress"):
        return 0, None  # a poll is already running; skip this tick rather than stack up
    st.session_state["_content_poll_in_progress"] = True
    try:
        offset = (
            st.session_state.get("content_last_update_id", 0) + 1
            if st.session_state.get("content_last_update_id") is not None
            else None
        )
        updates, err = get_telegram_updates(token, offset=offset)
        if err:
            return None, err
        messages = extract_messages(updates)
        for m in messages:
            st.session_state["content_last_update_id"] = max(
                st.session_state.get("content_last_update_id", 0) or 0, m["update_id"]
            )
            bucket = classify_message(m["text"], important_keywords)
            if bucket == "important":
                st.session_state.telegram_important.append(m)
            else:
                st.session_state.telegram_inbox.append(m)
            st.session_state.vectorstore = add_text_to_index(
                st.session_state.vectorstore,
                m["text"],
                source_label=f"Telegram Content Bot ({m['from']})",
            )
        return len(messages), None
    except Exception as e:
        return None, str(e)
    finally:
        st.session_state["_content_poll_in_progress"] = False


def answer_question(vectorstore, question, api_key):
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    source_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)
    return answer, source_docs


def answer_unified_query(vectorstore, question, api_key, state_context):
    """Like answer_question, but also feeds in the dashboard's current
    to-do/timetable/important-messages state, so questions from Telegram
    can span both documents and app state in one answer."""
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)

    if vectorstore is not None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        source_docs = retriever.invoke(question)
        doc_context = format_docs(source_docs)
    else:
        source_docs = []
        doc_context = "(No documents indexed yet.)"

    chain = COMBINED_PROMPT | llm | StrOutputParser()
    answer = chain.invoke(
        {"doc_context": doc_context, "state_context": state_context, "question": question}
    )
    return answer, source_docs


st.set_page_config(page_title="Multi-File RAG Dashboard", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: rgba(127, 127, 127, 0.05);
        border-radius: 12px;
        padding: 0.25rem 0.25rem 0.75rem 0.25rem;
    }
    .dash-card-title {
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .todo-done {
        text-decoration: line-through;
        opacity: 0.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 Multi-File RAG Dashboard")
st.caption(
    "Upload files (PDF, DOCX, TXT, MD, CSV, or audio), ask questions, message Telegram, "
    "and keep your schedule + to-dos all in one place."
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

# --- Telegram section state ---
if "telegram_inbox" not in st.session_state:
    st.session_state.telegram_inbox = []
if "telegram_important" not in st.session_state:
    st.session_state.telegram_important = []
if "telegram_last_update_id" not in st.session_state:
    st.session_state.telegram_last_update_id = None
if "telegram_keywords" not in st.session_state:
    st.session_state.telegram_keywords = "cricket, sports, research"
if "telegram_commands" not in st.session_state:
    st.session_state.telegram_commands = []  # log of {question, answer} pairs asked via Telegram

# --- Dashboard widgets state ---
if "timetable" not in st.session_state:
    st.session_state.timetable = []
if "todos" not in st.session_state:
    st.session_state.todos = []

with st.sidebar:
    st.header("1. Upload & index")
    uploaded_files = st.file_uploader(
        "Upload files (docs or audio)",
        type=["pdf", "docx", "txt", "md", "csv", "mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"],
        accept_multiple_files=True,
    )

    default_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    api_key = st.text_input(
        "GROQ API key (optional if set in Streamlit secrets)",
        type="password",
        value=default_key,
        key="groq_api_key",
    )
    st.caption("Audio files are transcribed with Groq's Whisper model before indexing, so a GROQ key is required for those.")

    st.divider()
    _worksheet, _log_error = get_log_worksheet()
    if _worksheet is not None:
        st.caption("📝 Logging: questions & answers are being saved to Google Sheets.")
    elif _log_error is None:
        st.caption("📝 Logging: off (add `gcp_service_account` and `GSHEET_URL` to secrets to enable).")
    else:
        st.caption(f"📝 Logging: off — connection failed: {_log_error}")

    if st.button("Build Index", type="primary", disabled=not uploaded_files):
        with st.spinner("Loading files, transcribing audio if any, and building index..."):
            try:
                vectorstore, msg, skipped = build_index(uploaded_files, api_key.strip())
                st.session_state.vectorstore = vectorstore
                if vectorstore is None:
                    st.error(msg)
                else:
                    st.success(msg)
                    if skipped:
                        st.warning(f"Skipped: {', '.join(skipped)}")
            except Exception as e:
                st.error(f"Something went wrong while building the index: {e}")

main_col, side_col = st.columns([2.4, 1], gap="large")

# ---------------------------------------------------------------------------
# MAIN COLUMN — RAG Q&A + Telegram
# ---------------------------------------------------------------------------
with main_col:
    st.header("2. Ask a question")
    col1, col2 = st.columns([3, 1])

    with col1:
        question = st.text_input("Type your question", value=st.session_state.question_text)

    with col2:
        st.write("Or record it:")
        voice_question = st.audio_input("Record a question")
        if voice_question is not None:
            st.audio(voice_question)  # play back what was actually captured, for sanity-checking
            audio_bytes = voice_question.getvalue()
            if not api_key.strip():
                st.warning("Add your GROQ API key to transcribe voice questions.")
            elif len(audio_bytes) < 8000:
                st.warning(
                    "That recording looks very short or silent (only "
                    f"{len(audio_bytes)} bytes). Whisper tends to hallucinate "
                    "\"Thank you\" on empty audio rather than erroring out. "
                    "Check your mic permissions/input device, then record again "
                    "and speak for a couple of seconds before stopping."
                )
            else:
                with st.spinner("Transcribing your question..."):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(voice_question.getbuffer())
                        tmp_path = tmp.name
                    question = transcribe_audio(tmp_path, api_key.strip())
                    os.remove(tmp_path)
                    st.session_state.question_text = question
                st.info(f"Heard: \"{question}\"")

    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked:
        if st.session_state.vectorstore is None:
            st.error("Upload files and click 'Build Index' first.")
        elif not question.strip():
            st.error("Please enter or record a question.")
        elif not api_key.strip():
            st.error("No GROQ API key found. Set it in the sidebar or in Streamlit secrets (GROQ_API_KEY).")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer, source_docs = answer_question(
                        st.session_state.vectorstore, question, api_key.strip()
                    )
                except Exception as e:
                    st.error(f"Something went wrong while answering: {e}")
                    answer, source_docs = None, []

            if answer is not None:
                log_qa(question, answer)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Sources")
                for i, d in enumerate(source_docs, 1):
                    src = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page")
                    loc = f"{src}" + (f" (page {page})" if page is not None else "")
                    with st.expander(f"[{i}] {loc}"):
                        st.write(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))

    # -----------------------------------------------------------------------
    # 3. Telegram
    # -----------------------------------------------------------------------
    st.divider()
    st.header("3. Telegram")
    st.markdown(
        "Your **main bot** answers questions — text it `/ask ...` (or start with `?`) and it "
        "answers using your documents, your todos/timetable, **and anything sent to it or your "
        "content bot below**, since those get embedded straight into the same RAG index."
    )

    default_bot_token = st.secrets.get("TELEGRAM_BOT_TOKEN", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    default_chat_id = st.secrets.get("TELEGRAM_CHAT_ID", os.environ.get("TELEGRAM_CHAT_ID", ""))

    tg_col1, tg_col2 = st.columns(2)
    with tg_col1:
        bot_token = st.text_input(
            "Main Bot Token",
            type="password",
            value=default_bot_token,
            help="Get this from @BotFather on Telegram.",
            key="main_bot_token",
        )
    with tg_col2:
        chat_id = st.text_input(
            "Chat ID",
            value=default_chat_id,
            help="The chat you're sending to / receiving from.",
            key="main_chat_id",
        )

    keywords_input = st.text_input(
        "Important keywords (comma-separated)",
        value=st.session_state.telegram_keywords,
        help="Any incoming message containing one of these words (case-insensitive) is filed under Important.",
    )
    st.session_state.telegram_keywords = keywords_input
    important_keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

    st.subheader("Send a message")
    send_col1, send_col2 = st.columns([4, 1])
    with send_col1:
        outgoing_text = st.text_area("Message", key="telegram_outgoing_text", height=80)
    with send_col2:
        st.write("")
        st.write("")
        send_clicked = st.button("Send", type="primary")

    if send_clicked:
        if not bot_token.strip() or not chat_id.strip():
            st.error("Add your Bot Token and Chat ID first.")
        elif not outgoing_text.strip():
            st.error("Write a message first.")
        else:
            ok, info = send_telegram_message(bot_token.strip(), chat_id.strip(), outgoing_text.strip())
            if ok:
                st.success("Message sent.")
            else:
                st.error(f"Failed to send: {info}")

    st.subheader("Incoming messages & questions")

    auto_col1, auto_col2 = st.columns([1, 1])
    with auto_col1:
        main_auto_poll = st.checkbox(
            "Auto-check for new messages",
            key="main_auto_poll",
            help="Polls automatically on a timer so /ask works without you opening the app.",
        )
    with auto_col2:
        main_poll_interval = st.number_input(
            "Every N seconds",
            min_value=10, max_value=300, value=10, step=5,
            key="main_poll_interval",
            disabled=not main_auto_poll,
        )

    check_col1, check_col2 = st.columns([1, 3])
    with check_col1:
        check_clicked = st.button("Check for new messages now")
    with check_col2:
        st.caption(
            "Messages starting with /ask, ?, /todos, /timetable, /important, /summary, or /help "
            "are answered directly back on Telegram. Everything else is embedded straight into "
            "your RAG index (no Inbox/Important sorting here — that's what the Content Bot below "
            "is for)."
        )

    if check_clicked:
        if not bot_token.strip():
            st.error("Add your Bot Token first.")
        else:
            answered, ingested_count, failures, poll_err = poll_main_bot_once(
                bot_token.strip(), chat_id.strip(), api_key
            )
            if poll_err:
                st.error(f"Couldn't fetch updates: {poll_err}")
            elif answered or ingested_count:
                st.success(
                    f"Answered {answered} command(s), ingested {ingested_count} message(s) "
                    "into your RAG index."
                )
                if failures:
                    st.error("Some messages failed:\n" + "\n".join(f"- {f}" for f in failures))
            else:
                st.info("No new messages.")

    if main_auto_poll:
        if not bot_token.strip():
            st.warning("Add your Main Bot Token above to enable auto-polling.")
        else:
            @st.fragment(run_every=main_poll_interval)
            def _main_bot_autopoll():
                token = st.session_state.get("main_bot_token", "").strip()
                default_chat = st.session_state.get("main_chat_id", "").strip()
                key = st.session_state.get("groq_api_key", "").strip()
                if not token:
                    return
                answered, ingested_count, failures, poll_err = poll_main_bot_once(
                    token, default_chat, key
                )
                ts = datetime.now().strftime("%H:%M:%S")
                if poll_err:
                    st.caption(f"⚠️ [{ts}] Poll failed: {poll_err}")
                elif failures:
                    st.caption(f"⚠️ [{ts}] Answered {answered}, ingested {ingested_count}, {len(failures)} failed.")
                else:
                    st.caption(f"🔄 [{ts}] Answered {answered} command(s), ingested {ingested_count} message(s).")

            _main_bot_autopoll()

        st.caption(
            "Note: like the Content Bot, this only polls while the app is open in a tab and awake "
            "(Streamlit Community Cloud's free tier sleeps idle apps). It's not a 24/7 background "
            "service — but as long as you keep a tab open, /ask will get answered without you "
            "touching the app."
        )

    with st.expander("➕ Content Bot (optional) — a second bot dedicated to feeding in content", expanded=True):
        st.markdown(
            "Useful if you want to keep 'notes/content' separate from 'questions' — e.g. forward "
            "articles, voice notes, or clippings to a **second bot**. Everything it receives is "
            "sorted into **Inbox** or **Important** (based on your keywords above) and embedded "
            "into the RAG index — no replies, pure intake. Create it the same way you made your "
            "main bot, via @BotFather."
        )
        default_content_bot_token = st.secrets.get(
            "TELEGRAM_CONTENT_BOT_TOKEN", os.environ.get("TELEGRAM_CONTENT_BOT_TOKEN", "")
        )
        content_bot_token = st.text_input(
            "Content Bot Token", type="password", value=default_content_bot_token, key="content_bot_token"
        )

        poll_col1, poll_col2 = st.columns([1, 1])
        with poll_col1:
            auto_poll = st.checkbox(
                "Auto-check for new content",
                key="content_auto_poll",
                help="Polls automatically on a timer while this tab stays open.",
            )
        with poll_col2:
            poll_interval = st.number_input(
                "Every N seconds",
                min_value=10, max_value=600, value=10, step=5,
                key="content_poll_interval",
                disabled=not auto_poll,
            )

        manual_clicked = st.button("Check for new content now")
        if manual_clicked:
            if not content_bot_token.strip():
                st.error("Add the Content Bot's token first.")
            else:
                count, err = poll_content_bot_once(content_bot_token.strip(), important_keywords)
                if err:
                    st.error(f"Couldn't fetch updates: {err}")
                elif count:
                    st.success(f"Sorted and ingested {count} message(s).")
                else:
                    st.info("No new content.")

        if auto_poll:
            if not content_bot_token.strip():
                st.warning("Add a Content Bot token above to enable auto-polling.")
            else:
                @st.fragment(run_every=poll_interval)
                def _content_bot_autopoll():
                    token = st.session_state.get("content_bot_token", "").strip()
                    if not token:
                        return
                    keywords = [
                        k.strip() for k in st.session_state.get("telegram_keywords", "").split(",") if k.strip()
                    ]
                    count, err = poll_content_bot_once(token, keywords)
                    ts = datetime.now().strftime("%H:%M:%S")
                    if err:
                        st.caption(f"⚠️ [{ts}] Poll failed: {err}")
                    else:
                        st.caption(f"🔄 [{ts}] Checked — sorted and ingested {count} new message(s).")

                _content_bot_autopoll()

            st.caption(
                "Note: auto-polling only runs while this app is open in a browser tab (and not "
                "asleep — e.g. Streamlit Community Cloud's free tier idles apps with no traffic). "
                "It's not a 24/7 background service; leave a tab open for it to keep working."
            )

    inbox_tab, important_tab, assistant_tab = st.tabs(
        [f"📥 Inbox ({len(st.session_state.telegram_inbox)})",
         f"⭐ Important ({len(st.session_state.telegram_important)})",
         f"🤖 Assistant ({len(st.session_state.telegram_commands)})"]
    )

    with inbox_tab:
        if not st.session_state.telegram_inbox:
            st.caption("No inbox messages yet.")
        else:
            for m in reversed(st.session_state.telegram_inbox):
                when = datetime.utcfromtimestamp(m["date"]).strftime("%Y-%m-%d %H:%M UTC") if m.get("date") else ""
                st.markdown(f"**{m['from']}** · {when}")
                st.write(m["text"])
                st.divider()

    with important_tab:
        if not st.session_state.telegram_important:
            st.caption("No important messages yet.")
        else:
            for m in reversed(st.session_state.telegram_important):
                when = datetime.utcfromtimestamp(m["date"]).strftime("%Y-%m-%d %H:%M UTC") if m.get("date") else ""
                st.markdown(f"**{m['from']}** · {when}")
                st.write(m["text"])
                st.divider()

    with assistant_tab:
        if not st.session_state.telegram_commands:
            st.caption("No questions asked via Telegram yet — try texting your bot `/ask ...`.")
        else:
            for qa in reversed(st.session_state.telegram_commands):
                when = (
                    datetime.utcfromtimestamp(qa["date"]).strftime("%Y-%m-%d %H:%M UTC")
                    if qa.get("date") else ""
                )
                st.markdown(f"**Q ({when}):** {qa['question']}")
                st.write(qa["answer"])
                st.divider()

# ---------------------------------------------------------------------------
# SIDE COLUMN — Timetable + To-Do List
# ---------------------------------------------------------------------------
with side_col:
    with st.container(border=True):
        st.markdown('<div class="dash-card-title">🗓️ Timetable</div>', unsafe_allow_html=True)

        with st.form("add_timetable_form", clear_on_submit=True):
            tt_time = st.time_input("Time", label_visibility="collapsed")
            tt_activity = st.text_input("Activity", placeholder="e.g. Team standup", label_visibility="collapsed")
            tt_submitted = st.form_submit_button("+ Add to timetable", use_container_width=True)
            if tt_submitted:
                add_timetable_entry(st.session_state, tt_time.strftime("%H:%M"), tt_activity)

        if not st.session_state.timetable:
            st.caption("Nothing scheduled yet.")
        else:
            for entry in st.session_state.timetable:
                row1, row2 = st.columns([5, 1])
                with row1:
                    st.markdown(f"**{entry['time']}** — {entry['activity']}")
                with row2:
                    if st.button("✕", key=f"del_tt_{entry['id']}", help="Remove"):
                        remove_timetable_entry(st.session_state, entry["id"])
                        st.rerun()

    with st.container(border=True):
        st.markdown('<div class="dash-card-title">✅ To-Do List</div>', unsafe_allow_html=True)

        with st.form("add_todo_form", clear_on_submit=True):
            todo_text = st.text_input("Task", placeholder="e.g. Review Q3 report", label_visibility="collapsed")
            todo_submitted = st.form_submit_button("+ Add task", use_container_width=True)
            if todo_submitted:
                add_todo(st.session_state, todo_text)

        if not st.session_state.todos:
            st.caption("No tasks yet.")
        else:
            pending = [t for t in st.session_state.todos if not t["done"]]
            done = [t for t in st.session_state.todos if t["done"]]

            for t in pending + done:
                row1, row2 = st.columns([5, 1])
                with row1:
                    checked = st.checkbox(t["text"], value=t["done"], key=f"todo_{t['id']}")
                    if checked != t["done"]:
                        toggle_todo(st.session_state, t["id"], checked)
                        st.rerun()
                with row2:
                    if st.button("✕", key=f"del_todo_{t['id']}", help="Remove"):
                        remove_todo(st.session_state, t["id"])
                        st.rerun()

            if done:
                st.caption(f"{len(done)}/{len(st.session_state.todos)} completed")
