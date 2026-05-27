import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()

COLLECTION_NAME = "rag_pdf"
QDRANT_URL = "http://localhost:6333"
TOP_K = 4

st.set_page_config(
    page_title="Postgres RAG",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-0: #0a0e1a;
        --bg-1: #111525;
        --bg-2: #161b2e;
        --bg-3: #1d2238;
        --border: rgba(148, 163, 184, 0.10);
        --border-hover: rgba(148, 163, 184, 0.22);
        --text-1: #e2e8f0;
        --text-2: #94a3b8;
        --text-3: #64748b;
        --accent-1: #8b5cf6;
        --accent-2: #6366f1;
        --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #3b82f6 100%);
        --success: #10b981;
    }

    html, body, .stApp, [class^="st-"], [class*=" st-"] {
        font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background: radial-gradient(ellipse at top, var(--bg-1) 0%, var(--bg-0) 100%);
        color: var(--text-1);
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { padding-top: 1.5rem; padding-bottom: 7rem; max-width: 920px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--bg-1);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

    /* Hero card */
    .hero {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.10) 0%, rgba(99, 102, 241, 0.05) 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    .hero p {
        margin: 0;
        color: var(--text-2);
        font-size: 0.98rem;
        line-height: 1.5;
    }

    /* Welcome / empty state */
    .welcome-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        margin: 1rem 0 2rem 0;
    }
    .welcome-card {
        background: var(--bg-2);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        transition: all 0.18s ease;
        cursor: pointer;
    }
    .welcome-card:hover {
        border-color: var(--border-hover);
        background: var(--bg-3);
        transform: translateY(-1px);
    }
    .welcome-card-label {
        color: var(--accent-1);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.35rem;
    }
    .welcome-card-text {
        color: var(--text-1);
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.35;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        gap: 0.85rem !important;
    }
    [data-testid="stChatMessage"] > div:first-child {
        background: var(--bg-3) !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stChatMessageContent"] {
        background: var(--bg-2);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 1.15rem !important;
        color: var(--text-1);
        font-size: 0.96rem;
        line-height: 1.6;
    }
    [data-testid="stChatMessage"][aria-label*="user"] [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.14), rgba(99, 102, 241, 0.10));
        border: 1px solid rgba(139, 92, 246, 0.25);
    }

    /* Markdown content inside chat messages */
    [data-testid="stChatMessageContent"] p {
        margin: 0 0 0.75rem 0;
        line-height: 1.65;
    }
    [data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0; }

    [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessageContent"] h3,
    [data-testid="stChatMessageContent"] h4 {
        color: var(--text-1);
        font-weight: 600;
        margin: 1.1rem 0 0.6rem 0;
        line-height: 1.3;
        letter-spacing: -0.01em;
    }
    [data-testid="stChatMessageContent"] h1 { font-size: 1.25rem; }
    [data-testid="stChatMessageContent"] h2 { font-size: 1.15rem; }
    [data-testid="stChatMessageContent"] h3 { font-size: 1.02rem; }
    [data-testid="stChatMessageContent"] h4 { font-size: 0.95rem; color: var(--text-2); }
    [data-testid="stChatMessageContent"] h1:first-child,
    [data-testid="stChatMessageContent"] h2:first-child,
    [data-testid="stChatMessageContent"] h3:first-child { margin-top: 0; }

    [data-testid="stChatMessageContent"] ul,
    [data-testid="stChatMessageContent"] ol {
        margin: 0.4rem 0 0.9rem 0;
        padding-left: 1.4rem;
    }
    [data-testid="stChatMessageContent"] li {
        margin: 0.3rem 0;
        line-height: 1.6;
    }
    [data-testid="stChatMessageContent"] li > p { margin: 0; }
    [data-testid="stChatMessageContent"] ul li::marker { color: var(--accent-1); }
    [data-testid="stChatMessageContent"] ol li::marker { color: var(--accent-1); font-weight: 600; }

    [data-testid="stChatMessageContent"] strong {
        color: #f1f5f9;
        font-weight: 600;
    }
    [data-testid="stChatMessageContent"] em {
        color: var(--text-1);
        font-style: italic;
    }

    [data-testid="stChatMessageContent"] code {
        background: rgba(139, 92, 246, 0.12);
        color: #c4b5fd;
        padding: 0.12rem 0.4rem;
        border-radius: 5px;
        font-size: 0.86em;
        font-family: 'JetBrains Mono', monospace;
        border: 1px solid rgba(139, 92, 246, 0.18);
    }
    [data-testid="stChatMessageContent"] pre {
        background: var(--bg-0);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.7rem 0 0.9rem 0;
        overflow-x: auto;
        font-size: 0.85rem;
        line-height: 1.55;
    }
    [data-testid="stChatMessageContent"] pre code {
        background: transparent;
        border: none;
        padding: 0;
        color: #e2e8f0;
        font-size: 0.85rem;
    }

    [data-testid="stChatMessageContent"] blockquote {
        border-left: 3px solid var(--accent-1);
        background: rgba(139, 92, 246, 0.06);
        padding: 0.6rem 1rem;
        margin: 0.7rem 0;
        border-radius: 0 8px 8px 0;
        color: var(--text-2);
    }
    [data-testid="stChatMessageContent"] blockquote p { margin: 0; }

    [data-testid="stChatMessageContent"] a {
        color: #a78bfa;
        text-decoration: none;
        border-bottom: 1px solid rgba(167, 139, 250, 0.35);
        transition: border-color 0.15s ease;
    }
    [data-testid="stChatMessageContent"] a:hover {
        border-bottom-color: #a78bfa;
    }

    [data-testid="stChatMessageContent"] table {
        border-collapse: collapse;
        margin: 0.7rem 0;
        font-size: 0.88rem;
        width: 100%;
    }
    [data-testid="stChatMessageContent"] th,
    [data-testid="stChatMessageContent"] td {
        border: 1px solid var(--border);
        padding: 0.4rem 0.7rem;
        text-align: left;
    }
    [data-testid="stChatMessageContent"] th {
        background: var(--bg-3);
        color: var(--text-1);
        font-weight: 600;
    }

    [data-testid="stChatMessageContent"] hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1rem 0;
    }

    /* Source cards */
    .source-wrap { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.4rem; }
    .source-card {
        background: var(--bg-2);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.75rem 0.95rem;
        transition: border-color 0.15s ease;
    }
    .source-card:hover { border-color: var(--border-hover); }
    .source-head {
        display: flex; align-items: center; gap: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .source-badge {
        background: var(--accent-gradient);
        color: white;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        letter-spacing: 0.04em;
    }
    .source-num {
        color: var(--text-3);
        font-size: 0.72rem;
        font-weight: 500;
    }
    .source-text {
        color: var(--text-2);
        font-size: 0.85rem;
        line-height: 1.5;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Sidebar content */
    .side-section {
        background: var(--bg-2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
    }
    .side-section h4 {
        margin: 0 0 0.65rem 0;
        color: var(--text-2);
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .side-row {
        display: flex; justify-content: space-between;
        align-items: center; padding: 0.25rem 0;
        font-size: 0.85rem;
    }
    .side-row .k { color: var(--text-3); }
    .side-row .v { color: var(--text-1); font-weight: 500; }
    .status-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 8px var(--success);
        margin-right: 0.4rem;
    }
    .brand {
        display: flex; align-items: center; gap: 0.65rem;
        padding: 0 0.25rem 1rem 0.25rem;
    }
    .brand-mark {
        width: 36px; height: 36px;
        background: var(--accent-gradient);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.35);
    }
    .brand-text { color: var(--text-1); font-weight: 600; font-size: 1.05rem; }
    .brand-sub { color: var(--text-3); font-size: 0.72rem; }

    /* Buttons */
    .stButton > button {
        background: var(--bg-2);
        color: var(--text-1);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.5rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: var(--bg-3);
        border-color: var(--border-hover);
        color: var(--text-1);
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: transparent !important;
        border-top: 1px solid var(--border);
        backdrop-filter: blur(12px);
    }
    [data-testid="stChatInput"] textarea {
        background: var(--bg-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        color: var(--text-1) !important;
        font-size: 0.95rem !important;
        padding: 0.85rem 1.1rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--accent-1) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.12) !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-top: 0.5rem;
    }
    [data-testid="stExpander"] summary {
        font-size: 0.82rem;
        color: var(--text-2);
        padding: 0.5rem 0.9rem;
    }

    /* Spinner / status */
    [data-testid="stStatusWidget"] { display: none; }
    .stSpinner > div { border-top-color: var(--accent-1) !important; }

    /* Error block */
    .err-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(220, 38, 38, 0.04));
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        color: var(--text-1);
    }
    .err-card h3 { margin: 0 0 0.5rem 0; color: #fca5a5; font-size: 1rem; font-weight: 600; }
    .err-card p { margin: 0.3rem 0; color: var(--text-2); font-size: 0.9rem; }
    .err-card code { background: var(--bg-3); padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.82rem; }

    /* Divider */
    hr { border-color: var(--border) !important; margin: 0.75rem 0 !important; }

    /* Toggle */
    [data-testid="stWidgetLabel"] { color: var(--text-2) !important; font-size: 0.82rem !important; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_vector_store():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    return QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.2)


@st.cache_data(ttl=30, show_spinner=False)
def get_collection_stats():
    import requests
    try:
        r = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}", timeout=2)
        if r.ok:
            return r.json()["result"]["points_count"]
    except Exception:
        pass
    return None


SAMPLE_QUERIES = [
    ("Quickstart", "What is PostgreSQL and what is it used for?"),
    ("Language", "Explain PL/pgSQL and when to use it"),
    ("Triggers", "How do trigger functions work?"),
    ("Transactions", "How does transaction management work in PL/pgSQL?"),
]

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">P</div>
            <div>
                <div class="brand-text">Postgres RAG</div>
                <div class="brand-sub">Knowledge assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    points = get_collection_stats()
    status_label = (
        f'<span class="status-dot"></span>Online &middot; {points:,} chunks'
        if points is not None
        else '<span class="status-dot" style="background:#ef4444; box-shadow:0 0 8px #ef4444;"></span>Qdrant offline'
    )
    st.markdown(
        f"""
        <div class="side-section">
            <h4>Status</h4>
            <div style="color: var(--text-1); font-size: 0.88rem;">{status_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-section">
            <h4>Stack</h4>
            <div class="side-row"><span class="k">Embeddings</span><span class="v">nomic-embed-text (local)</span></div>
            <div class="side-row"><span class="k">LLM</span><span class="v">gemini-flash-latest</span></div>
            <div class="side-row"><span class="k">Vector DB</span><span class="v">Qdrant</span></div>
            <div class="side-row"><span class="k">Top-K</span><span class="v">4 chunks</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_query", None)
        st.rerun()


st.markdown(
    """
    <div class="hero">
        <h1>Ask the PostgreSQL Docs</h1>
        <p>A retrieval-augmented chat experience grounded in the official PostgreSQL 18.4 documentation. Every answer is backed by citable source chunks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    vector_store = get_vector_store()
    llm = get_llm()
except Exception as e:
    st.markdown(
        f"""
        <div class="err-card">
            <h3>Backend not reachable</h3>
            <p>Make sure Docker is running and Qdrant is up:</p>
            <p><code>docker compose up -d</code></p>
            <p>Also confirm <code>GOOGLE_API_KEY</code> is set in <code>.env</code>.</p>
            <p style="margin-top:0.75rem; font-size:0.78rem; color: var(--text-3);">{e}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    cards = ""
    for i, src in enumerate(sources, 1):
        page = src.get("page", "?")
        page_display = f"Page {page}" if page != "?" else "Unknown"
        cards += f"""
        <div class="source-card">
            <div class="source-head">
                <span class="source-badge">{page_display}</span>
                <span class="source-num">chunk {i} of {len(sources)}</span>
            </div>
            <div class="source-text">{src['content']}</div>
        </div>
        """
    st.markdown(f'<div class="source-wrap">{cards}</div>', unsafe_allow_html=True)


def run_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = vector_store.similarity_search(query, k=TOP_K)
            context = "\n\n".join(doc.page_content for doc in results)
            system_prompt = (
                "You are a helpful, conversational assistant. Answer the user's "
                "question naturally, in flowing paragraphs, the way ChatGPT would. "
                "Base your answer strictly on the context below; if the answer is "
                "not in the context, say you don't know. Do not mention the "
                "context, sources, page numbers, or that you were given documents. "
                "Just answer the question directly and clearly.\n\n"
                f"Context:\n{context}"
            )
            response = llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ]
            )
            raw = response.content
            if isinstance(raw, str):
                answer = raw
            elif isinstance(raw, list):
                answer = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in raw
                )
            else:
                answer = str(raw)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if not st.session_state.messages:
    st.markdown(
        '<div style="color: var(--text-3); font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin: 0.5rem 0 0.75rem 0;">Try one of these</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, (label, q) in enumerate(SAMPLE_QUERIES):
        with cols[i % 2]:
            if st.button(
                f"{label}  ·  {q}",
                key=f"sample_{i}",
                use_container_width=True,
            ):
                st.session_state.pending_query = q
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


pending = st.session_state.pop("pending_query", None)
typed = st.chat_input("Ask anything about PostgreSQL...")
query = pending or typed

if query:
    run_query(query)
