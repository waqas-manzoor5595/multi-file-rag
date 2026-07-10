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


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_log_worksheet():
    """Connect to the Google Sheet used for logging Q&A. Returns None if not configured."""
    try:
        service_account_info = st.secrets["gcp_service_account"]
        sheet_url = st.secrets["GSHEET_URL"]
    except Exception:
        return None

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
    except Exception:
        return None

    return worksheet


def log_qa(question, answer):
    """Best-effort logging: never let a logging failure break the app."""
    worksheet = get_log_worksheet()
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


def load_any_file(path, display_name, api_key=None):
    ext = os.path.splitext(display_name)[1].lower()

    if ext in AUDIO_EXTENSIONS:
        if not api_key:
            return []
        text = transcribe_audio(path, api_key)
        if not text or not text.strip():
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
            all_docs.extend(load_any_file(tmp_path, uf.name, api_key))

    if not all_docs:
        return None, "Couldn't extract text from any uploaded file.", skipped

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    indexed_count = len(uploaded_files) - len(skipped)
    msg = f"Indexed {len(chunks)} chunks from {indexed_count} file(s)."
    return vectorstore, msg, skipped


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


st.set_page_config(page_title="Multi-File RAG", page_icon="📚", layout="wide")
st.title("📚 Multi-File RAG")
st.markdown(
    "Upload files (PDF, DOCX, TXT, MD, CSV, or **audio**), build the index, then ask "
    "questions — by typing or by voice. Answers cite which file they came from."
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

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
    )
    st.caption("Audio files are transcribed with Groq's Whisper model before indexing, so a GROQ key is required for those.")

    st.divider()
    if get_log_worksheet() is not None:
        st.caption("📝 Logging: questions & answers are being saved to Google Sheets.")
    else:
        st.caption("📝 Logging: off (add `gcp_service_account` and `GSHEET_URL` to secrets to enable).")

    if st.button("Build Index", type="primary", disabled=not uploaded_files):
        with st.spinner("Loading files, transcribing audio if any, and building index..."):
            vectorstore, msg, skipped = build_index(uploaded_files, api_key.strip())
            st.session_state.vectorstore = vectorstore
            if vectorstore is None:
                st.error(msg)
            else:
                st.success(msg)
                if skipped:
                    st.warning(f"Skipped: {', '.join(skipped)}")

st.header("2. Ask a question")

col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Type your question", value=st.session_state.question_text)
with col2:
    st.write("Or record it:")
    voice_question = st.audio_input("Record a question")

if voice_question is not None:
    if not api_key.strip():
        st.warning("Add your GROQ API key to transcribe voice questions.")
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
            answer, source_docs = answer_question(
                st.session_state.vectorstore, question, api_key.strip()
            )
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
