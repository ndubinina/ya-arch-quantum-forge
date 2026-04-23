from rag_bot import ask_rag
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime
import json

emb = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    encode_kwargs={"normalize_embeddings": True}
)

def load_questions():
    tests = []
    with open("data/golden_questions.txt", encoding="utf-8") as f:
        block = {}
        for line in f:
            line = line.strip()
            if not line:
                if block:
                    tests.append(block)
                    block = {}
                continue
            if line.startswith("Вопрос:"):
                block["question"] = line.replace("Вопрос:", "").strip()

            elif line.startswith("Ответ:"):
                block["expected"] = line.replace("Ответ:", "").strip()
        if block:
            tests.append(block)
    return tests

def similarity(a, b):
    va = emb.embed_query(a)
    vb = emb.embed_query(b)
    return np.dot(va, vb)

def run_tests():
    questions = load_questions()
    print(questions[0])
    results = []
    for q in questions:
        answer, docs = ask_rag(q["question"])
        similar = similarity(answer, q["expected"])
        result = {
            "question": q["question"],
            "similarity": similar,
            "expected": q["expected"],
            "answer": answer,
            "answer_is_found": "я не знаю" not in answer.lower(),
            "answer_is_similarity": "true" if similar > 0.8 else "false",
            "chunk_size": len(docs),
            "sources": [
                {
                    "metadata": doc.metadata,
                    "content": doc.page_content[:800]
                }
                for doc in docs
            ],
        }
        print(result)
        log(result)
    return


LOG_FILE = "data/logs.jsonl"

def log(
    result,
):
    log = {
        "result": result,
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    run_tests()