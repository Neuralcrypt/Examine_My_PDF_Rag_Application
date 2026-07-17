import html
import os

from dotenv import find_dotenv, load_dotenv
import httpx
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from database import (
    build_vectorstore,
    get_retriever,
    release_vectorstore,
    reset_app_storage,
    save_uploaded_pdf,
)


load_dotenv(find_dotenv(usecwd=True), override=True)

APP_NAME = "Examine my pdf"
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant. Use only the provided context to answer the question. If the answer is not present in the context, say: I could not find the answer in the document.""",
        ),
        (
            "human",
            """Context:\n{context}\n\nQuestion:\n{question}""",
        ),
    ]
)


def get_mistral_api_key() -> str:
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        secret_value = st.secrets.get("MISTRAL_API_KEY", "")
    except Exception:
        secret_value = ""

    api_key = str(secret_value).strip()
    if api_key:
        return api_key

    raise RuntimeError(
        "Missing MISTRAL_API_KEY. Add it to .env for local runs or Streamlit secrets for deployment, then restart the app."
    )


def create_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-small-2506",
        api_key=get_mistral_api_key(),
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 196, 146, 0.32), transparent 28%),
                radial-gradient(circle at top right, rgba(113, 201, 206, 0.24), transparent 22%),
                linear-gradient(180deg, #fffaf3 0%, #fff4e6 48%, #f8f4ef 100%);
            color: #2f241d;
        }
        [data-testid="stHeader"] {
            background: rgba(255, 250, 243, 0.8);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f3a44 0%, #2d5a63 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #f5efe6;
        }
        [data-testid="stFileUploaderDropzone"],
        [data-testid="stFileUploaderDropzone"] * {
            color: #2f241d !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(255, 250, 243, 0.96);
            border: 1px solid rgba(188, 142, 100, 0.24);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 8rem;
            max-width: 1000px;
        }
        .hero-card {
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,241,224,0.86));
            border: 1px solid rgba(212, 166, 121, 0.24);
            box-shadow: 0 18px 50px rgba(122, 84, 45, 0.12);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.45rem;
            letter-spacing: -0.02em;
        }
        .hero-copy {
            font-size: 1.02rem;
            line-height: 1.65;
            color: #5c4737;
            margin-bottom: 0;
        }
        .feature-row {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin: 1rem 0 1.2rem 0;
        }
        .feature-pill {
            padding: 0.55rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(188, 142, 100, 0.2);
            font-size: 0.92rem;
            color: #5a4331;
        }
        .empty-state {
            padding: 1.4rem 1.2rem;
            border-radius: 20px;
            background: rgba(255,255,255,0.72);
            border: 1px dashed rgba(124, 91, 62, 0.24);
            color: #5f4a3b;
            margin-top: 0.5rem;
        }
        .stats-card {
            padding: 1rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(188, 142, 100, 0.2);
            text-align: center;
        }
        .stats-number {
            font-size: 1.55rem;
            font-weight: 800;
            color: #1f3a44;
        }
        .stats-label {
            font-size: 0.9rem;
            color: #6d5645;
        }
        .stButton > button {
            width: 100%;
            border-radius: 14px;
            border: 0;
            padding: 0.75rem 1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f28c54 0%, #db6f37 100%);
            color: white;
            box-shadow: 0 10px 24px rgba(219, 111, 55, 0.28);
        }
        div[data-testid="stBottomBlockContainer"] {
            background: transparent;
        }
        div[data-testid="stBottomBlockContainer"] > div {
            max-width: 920px;
            margin: 0 auto 1rem auto;
        }
        .stChatFloatingInputContainer,
        div[data-testid="stChatInput"] {
            background: rgba(255, 252, 247, 0.94);
            border: 1px solid rgba(188, 142, 100, 0.24);
            border-radius: 26px;
            box-shadow: 0 18px 40px rgba(122, 84, 45, 0.16);
            backdrop-filter: blur(12px);
        }
        div[data-testid="stChatInput"] {
            padding: 0.35rem 0.55rem;
        }
        div[data-testid="stChatInput"] textarea {
            color: #2f241d;
        }
        .message-bubble {
            padding: 0.95rem 1.1rem;
            border-radius: 22px;
            line-height: 1.6;
            font-size: 0.98rem;
            box-shadow: 0 14px 30px rgba(122, 84, 45, 0.08);
            margin-bottom: 0.55rem;
            word-break: break-word;
        }
        .message-bubble.user {
            background: linear-gradient(135deg, #ffffff 0%, #fff0de 100%);
            border: 1px solid rgba(219, 111, 55, 0.18);
            color: #4f392a;
            border-bottom-right-radius: 8px;
        }
        .message-bubble.assistant {
            background: linear-gradient(135deg, #1f3a44 0%, #2f5962 100%);
            border: 1px solid rgba(31, 58, 68, 0.2);
            color: #f8f4ef;
            border-bottom-left-radius: 8px;
        }
        .message-wrap {
            margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "app_initialized" not in st.session_state:
        reset_app_storage()
        st.session_state.app_initialized = True
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.vectorstore = None
        st.session_state.active_pdf_name = None
        st.session_state.page_count = 0
        st.session_state.chunk_count = 0
        st.session_state.chat_ready = False
        st.session_state.llm = create_llm()


def clear_chatbot() -> None:
    vectorstore = st.session_state.get("vectorstore")
    st.session_state.retriever = None
    st.session_state.vectorstore = None
    release_vectorstore(vectorstore)
    reset_app_storage()

    for key in [
        "app_initialized",
        "messages",
        "retriever",
        "vectorstore",
        "active_pdf_name",
        "page_count",
        "chunk_count",
        "chat_ready",
        "llm",
    ]:
        st.session_state.pop(key, None)
    st.rerun()


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Examine my pdf</div>
            <p class="hero-copy">Upload one PDF, ask natural questions, and get answers grounded only in that document. The experience is built like a real chat app, but keeps the focus tightly on your file.</p>
            <div class="feature-row">
                <div class="feature-pill">Single-document RAG</div>
                <div class="feature-pill">Floating composer</div>
                <div class="feature-pill">Answers only from PDF context</div>
                <div class="feature-pill">Fresh database on each new session</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats() -> None:
    if not st.session_state.chat_ready:
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="stats-card">
                <div class="stats-number">1</div>
                <div class="stats-label">Active PDF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="stats-card">
                <div class="stats-number">{st.session_state.page_count}</div>
                <div class="stats-label">Pages Loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="stats-card">
                <div class="stats-number">{st.session_state.chunk_count}</div>
                <div class="stats-label">Search Chunks</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Control Panel")
        st.write("Add a PDF, build the knowledge base, and then chat with it below.")

        uploaded_pdf = st.file_uploader(
            "Upload your PDF",
            type=["pdf"],
            help="Only one PDF is used at a time for this chatbot session.",
        )

        if uploaded_pdf is not None and st.button("Prepare PDF"):
            with st.spinner("Reading the PDF and creating the search database. Have some patience, this may take a few seconds depending on the size of the PDF..."):
                existing_vectorstore = st.session_state.get("vectorstore")
                st.session_state.retriever = None
                st.session_state.vectorstore = None
                release_vectorstore(existing_vectorstore)

                pdf_path = save_uploaded_pdf(uploaded_pdf)
                vectorstore, page_count, chunk_count = build_vectorstore(pdf_path)
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = get_retriever(vectorstore)
                st.session_state.active_pdf_name = uploaded_pdf.name
                st.session_state.page_count = page_count
                st.session_state.chunk_count = chunk_count
                st.session_state.chat_ready = True
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": f"I have finished reading {uploaded_pdf.name}. Ask me anything from this PDF, and I will answer only from its content.",
                        "sources": [],
                    }
                ]
            st.rerun()

        if st.session_state.chat_ready:
            st.markdown("---")
            st.write(f"Current PDF: **{st.session_state.active_pdf_name}**")
            st.write("Close and reopen the chatbot for an automatic fresh start, or use the reset button now.")
            if st.button("Delete database and start fresh"):
                clear_chatbot()


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            Upload a PDF from the sidebar to begin. Once the file is processed, this area turns into a live document chat where each answer is based only on the uploaded PDF.
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_sources(docs):
    sources = []
    for doc in docs:
        page_number = doc.metadata.get("page")
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 220:
            snippet = f"{snippet[:220]}..."
        sources.append(
            {
                "page": "Unknown" if page_number is None else page_number + 1,
                "snippet": snippet,
            }
        )
    return sources


def answer_question(user_query: str):
    docs = st.session_state.retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke(
        {
            "context": context,
            "question": user_query,
        }
    )
    response = st.session_state.llm.invoke(final_prompt)
    return response.content, build_sources(docs)


def show_auth_help(error_message: str) -> None:
    st.error(error_message)
    st.info(
        "If you added or changed the API key after the app was already running, stop the server and restart it with `.venv\\Scripts\\python.exe -m streamlit run main.py`."
    )


def format_bubble_content(content: str) -> str:
    return html.escape(content).replace("\n", "<br>")


def render_message(message: dict) -> None:
    role = message["role"]
    content = format_bubble_content(message["content"])

    if role == "user":
        left, right = st.columns([0.28, 0.72])
        with right:
            st.markdown(
                f"<div class='message-wrap'><div class='message-bubble user'>{content}</div></div>",
                unsafe_allow_html=True,
            )
    else:
        left, right = st.columns([0.72, 0.28])
        with left:
            st.markdown(
                f"<div class='message-wrap'><div class='message-bubble assistant'>{content}</div></div>",
                unsafe_allow_html=True,
            )
            if message.get("sources"):
                with st.expander("View supporting passages"):
                    for index, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**Source {index} | Page {source['page']}**")
                        st.caption(source["snippet"])


st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
)

apply_theme()

try:
    initialize_state()
except RuntimeError as error:
    render_header()
    show_auth_help(str(error))
    st.stop()

render_sidebar()
render_header()
render_stats()

if not st.session_state.chat_ready:
    render_empty_state()
else:
    for message in st.session_state.messages:
        render_message(message)

    user_query = st.chat_input("Ask something from the uploaded PDF...")
    if user_query:
        user_message = {
            "role": "user",
            "content": user_query,
        }
        st.session_state.messages.append(user_message)
        render_message(user_message)

        try:
            with st.spinner("Searching the PDF and drafting an answer..."):
                answer, sources = answer_question(user_query)

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
            st.session_state.messages.append(assistant_message)
            render_message(assistant_message)
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 401:
                show_auth_help(
                    "Mistral rejected the API key with a 401 Unauthorized response."
                )
            else:
                st.error(f"Request failed: {error}")

