import os
import sys
import streamlit as st

# Force Python to use pysqlite3 instead of old sqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # pysqlite3 not installed, fallback to default


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


load_dotenv()

CHUNK_SIZE=1000
EMBEDDING_MODEL="Alibaba-NLP/gte-base-en-v1.5"
VECTOR_STORE_DIR=Path(__file__).parent/"resources/vector_store"
COLLECTION_NAME="real_estate"
# hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

# Ensure token is set
if not hf_token:
    raise ValueError("Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN in .env or system environment.")

# Authenticate before using the model
login(token=hf_token)

llm=None
vector_store=None

def initialize_components():
    global llm,vector_store

    if llm is None:
        llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef= HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code":True}
        )

        vector_store=Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTOR_STORE_DIR)

        )


def process_urls(urls):
    """
    This function scraps data from the urls and stores it in a vector db
    :param urls: input urls
    :return
    """

    # print("Initialize components")
    yield "Initializing components...✅"
    initialize_components()
    vector_store.reset_collection()

    # print("load data")
    yield "Resetting vector store...✅"
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()

    # print("Split Text")
    yield "Splitting text into chunks...✅"
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size=CHUNK_SIZE
    )


    docs=text_splitter.split_documents(data)

    # print("Add docs to vector db")
    yield "Adding chunks to vector database...✅"
    uuids=[str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)
   
    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector Database is not initialized")
   
    chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vector_store.as_retriever())
    result=chain.invoke({"question":query}, return_only_outputs=True)
    sources=result.get("sources","")

    return result['answer'],sources

if __name__=="__main__":
    urls=[
        # "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html&quot;,
        # "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html&quot;
        "https://www.foxbusiness.com/personal-finance/todays-mortgage-rates-august-14-2024",
        "https://www.foxbusiness.com/personal-finance/todays-mortgage-rates-august-13-2024"
    ]  #This will process the urls

    process_urls(urls)

    # results=vector_store.similarity_search(
    #     "30 year mortgage rate",
    #     k=2
    # )
    # print(results)

    answer,sources=generate_answer("Tell me what was the 30 year fixed mortgate rate along with the date?")
    print(f"Answer:{answer}")
    print(f"sources:{sources}")