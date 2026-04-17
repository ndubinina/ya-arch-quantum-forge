import logging
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_PATH = Path("docs")
LOG_FILE = "logs/update.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def updateIndex(file):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = text_splitter.split_documents(TextLoader(str(file), encoding="utf-8").load())
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content.lower()}"

    logging.info(f"Chunks created: {len(chunks)}")


    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={"normalize_embeddings": True},
    )

    db = Chroma(
        persist_directory="chroma_index",
        embedding_function=embedding_model,
    )
    db.add_documents(chunks)
    db.persist()
    return

logging.info("Start update index")
for file in DOCS_PATH.glob("*.txt"):
    try:
        logging.info(f"New file detected: {file.name}")
        processing = file.with_suffix(".processing")
        file.rename(processing)
        updateIndex(processing)
        processing.rename(processing.with_suffix(".processed"))
        logging.info(f"{file.name} was processed")
    except Exception as e:
        logging.exception(f"Failed processing {file.name}: {e}")
logging.info("Index updated successfully.")
