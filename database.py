from pathlib import Path
import gc
import shutil
import time

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
APP_DATA_DIR = BASE_DIR / ".app_data"
UPLOAD_DIR = APP_DATA_DIR / "uploads"
CHROMA_DIR = APP_DATA_DIR / "chroma_db"


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )


def release_vectorstore(vectorstore) -> None:
    if vectorstore is None:
        return

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    try:
        client = getattr(vectorstore, "_client", None)
        system = getattr(client, "_system", None)
        if system is not None:
            system.stop()
    except Exception:
        pass

    gc.collect()
    time.sleep(0.25)


def remove_tree_with_retries(path: Path, retries: int = 6, delay: float = 0.25) -> None:
    if not path.exists():
        return

    last_error = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as error:
            last_error = error
            gc.collect()
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

    if last_error is not None:
        raise last_error


def reset_app_storage() -> None:
    remove_tree_with_retries(APP_DATA_DIR)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_pdf(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = UPLOAD_DIR / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getbuffer())
    return pdf_path


def load_pdf_documents(pdf_path: Path):
    loader = PyMuPDFLoader(str(pdf_path))
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(documents)


def build_vectorstore(pdf_path: Path):
    remove_tree_with_retries(CHROMA_DIR)

    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore, len(documents), len(chunks)


def get_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )
