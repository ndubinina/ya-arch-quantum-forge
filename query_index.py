from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

db = Chroma(
    persist_directory="chroma_index",
    embedding_function=embeddings
)

def question(query):
    query = f"query: {query}"
    results = db.similarity_search_with_score(query, k=3)

    print(query)
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1}:")
        print(doc.page_content[:1000])
        print(doc.metadata)
        print(score)

def main():
    print()
    question("Кто главный герой цикла книг?")
    print()
    question("Какие сильные стороны Ивана?")
    print()
    question("Какие факультеты есть в школе Поваргвардс?")
    print()
    question("Какого цвета распределяющая фольга?")

if __name__ == "__main__":
    main()