import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
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


def load_any_file(path, display_name):
    ext = os.path.splitext(display_name)[1].lower()
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


def build_index(uploaded_files):
    all_docs = []
    skipped = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uf in uploaded_files:
            ext = os.path.splitext(uf.name)[1].lower()
            if ext not in LOADER_MAP:
                skipped.append(uf.name)
                continue
            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())
            all_docs.extend(load_any_file(tmp_path, uf.name))

    if not all_docs:
        return None, "Couldn't extract text from any uploaded file.", skipped

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    msg = f"Indexed {len(chunks)} chunks from {len(uploaded_files) - len(skipped)} file(s)."
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
    "Upload one or more files (PDF, DOCX, TXT, MD, CSV), build the index, then ask "
    "questions across all of them. Answers cite which file they came from."
)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.header("1. Upload & index")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
    )
    default_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    api_key = st.text_input(
        "GROQ API key (optional if set in Streamlit secrets)",
        type="password",
        value=default_key,
    )

    if st.button("Build Index", type="primary", disabled=not uploaded_files):
        with st.spinner("Loading files and building index..."):
            vectorstore, msg, skipped = build_index(uploaded_files)
            st.session_state.vectorstore = vectorstore
            if vectorstore is None:
                st.error(msg)
            else:
                st.success(msg)
                if skipped:
                    st.warning(f"Skipped unsupported: {', '.join(skipped)}")

st.header("2. Ask a question")
question = st.text_input("Your question")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if st.session_state.vectorstore is None:
        st.error("Upload files and click 'Build Index' first.")
    elif not question.strip():
        st.error("Please enter a question.")
    elif not api_key.strip():
        st.error("No GROQ API key found. Set it in the sidebar or in Streamlit secrets (GROQ_API_KEY).")
    else:
        with st.spinner("Thinking..."):
            answer, source_docs = answer_question(
                st.session_state.vectorstore, question, api_key.strip()
            )
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, d in enumerate(source_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            loc = f"{src}" + (f" (page {page})" if page is not None else "")
            with st.expander(f"[{i}] {loc}"):
                st.write(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
