import time
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = Path(__file__).parent / "ragpdf3.pdf"
COLLECTION_NAME = "rag_pdf"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
PAGES_PER_BATCH = 20
MAX_PAGES = 300  # cap at first 300 pages of the PDF

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
embedding_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
loader = PyPDFLoader(str(PDF_PATH))

vector_store = None
page_buffer = []
total_pages = 0
total_chunks = 0


def flush_batch():
    """Embed the buffered pages and push them to Qdrant, retrying on transient errors."""
    global vector_store, page_buffer, total_chunks
    if not page_buffer:
        return
    chunks = splitter.split_documents(page_buffer)

    attempt = 0
    while True:
        try:
            if vector_store is None:
                vector_store = QdrantVectorStore.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    url=QDRANT_URL,
                    collection_name=COLLECTION_NAME,
                )
            else:
                vector_store.add_documents(chunks)
            break
        except Exception as e:
            msg = str(e)
            is_network = (
                "getaddrinfo failed" in msg
                or "ConnectError" in msg
                or "ConnectionError" in msg
                or "Connection reset" in msg
                or "timed out" in msg.lower()
            )
            if is_network:
                attempt += 1
                if attempt > 5:
                    print(f"  Giving up after {attempt} attempts: {msg[:200]}", flush=True)
                    raise
                wait = min(30, 5 * attempt)
                print(f"  Network error, sleeping {wait}s (attempt {attempt})...", flush=True)
                time.sleep(wait)
                continue
            raise

    total_chunks += len(chunks)
    page_buffer = []
    print(
        f"Pages: {total_pages:>5}  |  Chunks uploaded: {total_chunks:>6}",
        flush=True,
    )


print(f"Starting Ollama ingestion (first {MAX_PAGES} pages, model={EMBED_MODEL})...", flush=True)

for page in loader.lazy_load():
    page_buffer.append(page)
    total_pages += 1
    if len(page_buffer) >= PAGES_PER_BATCH:
        flush_batch()
    if total_pages >= MAX_PAGES:
        break

flush_batch()  # leftover pages

print(
    f"\nDone. {total_pages} pages -> {total_chunks} chunks in '{COLLECTION_NAME}'.",
    flush=True,
)
