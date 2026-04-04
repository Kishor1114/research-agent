import datetime


def save_to_memory(collection, embedder, question, answer, confidence="medium"):
    text = f"Q: {question}\nA: {answer}"
    embedding = embedder.encode(text).tolist()
    existing = collection.count()
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "confidence": confidence,
        "question": question[:100]
    }
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[f"mem_{existing + 1}"],
        metadatas=[metadata]
    )


def check_memory(collection, embedder, question, threshold=0.8):
    if collection.count() == 0:
        return None
    embedding = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=1)
    distance = results["distances"][0][0]
    if distance < threshold:
        return results["documents"][0][0]
    return None


def get_memory_stats(collection):
    count = collection.count()
    if count == 0:
        return {"count": 0, "topics": []}
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        high_conf = sum(1 for m in metadatas if m.get("confidence") == "high")
        return {
            "count": count,
            "high_confidence": high_conf,
            "hit_rate": round(high_conf / count * 100, 1) if count > 0 else 0
        }
    except:
        return {"count": count, "high_confidence": 0, "hit_rate": 0}