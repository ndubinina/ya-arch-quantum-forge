import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import time

DATA_PATH = "knowledge_base"

documents = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print(f"Loaded documents: {len(documents)}")

start = time.time()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)
for chunk in chunks:
    chunk.page_content = f"passage: {chunk.page_content.lower()}"

print(f"Chunks created: {len(chunks)}")

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_index"
)

print(f"creation vectorestore took {time.time() - start:.2f} seconds")

vectorstore.persist()
print("ChromaDB index saved in 'chroma_index'")