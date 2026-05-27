from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

load_dotenv()

embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

# Connect to the existing Qdrant collection (created by index.py)
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="rag_pdf",
)

# Take a user query
query = input("Ask a question: ")

# Retrieve the top 3 most relevant chunks
results = vector_store.similarity_search(query, k=3)

# Build context from the retrieved chunks
context = "\n\n".join(
    f"[Page {doc.metadata.get('page')}]\n{doc.page_content}" for doc in results
)

system_prompt = f"""You are a helpful assistant that answers questions strictly
based on the context below. If the answer is not in the context, say you don't know.

Context:
{context}
"""

# Ask the LLM
response = llm.invoke([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": query},
])

raw = response.content
if isinstance(raw, str):
    answer = raw
elif isinstance(raw, list):
    answer = "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in raw
    )
else:
    answer = str(raw)

print("\n--- Answer ---")
print(answer)
