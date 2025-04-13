from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from huggingface_hub import login
import os
import sys

# Attempt to use pysqlite3 if available
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # fallback to default sqlite3 if pysqlite3 is not available

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
COLLECTION_NAME = "real_estate"
VECTOR_STORE_DIR = Path(__file__).parent / "resource/vector_store"

# Load HuggingFace Token
hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN not found in environment variables.")
login(token=hf_token)

# Global variables
llm = None
vector_store = None

def initialize_components():
    """Initializes LLM and vector store if not already initialized."""
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STORE_DIR)
        )

def process_urls(urls):
    """
    Scrapes data from URLs, splits into chunks, and stores in vector DB.
    :param urls: List of URLs
    :yield: Status messages for Streamlit
    """
    yield "🔧 Initializing components..."
    initialize_components()
    vector_store.reset_collection()

    yield "🗑️ Resetting vector store..."
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "✂️ Splitting data into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "📦 Adding chunks to vector database..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "✅ Done adding docs to vector database."

def generate_answer(query):
    """
    Answers a query based on the processed vector DB.
    :param query: User question
    :return: Tuple of (answer, sources)
    """
    if not vector_store:
        raise RuntimeError("Vector store not initialized.")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")
    return result["answer"], sources

if __name__ == "__main__":
    urls = [
        "https://www.foxbusiness.com/personal-finance/todays-mortgage-rates-august-14-2024",
        "https://www.foxbusiness.com/personal-finance/todays-mortgage-rates-august-13-2024"
    ]
    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer("Tell me about the mortgage rates")
    print("\nAnswer:", answer)
    print("Sources:", sources)
