from groq import Groq
import requests
import chromadb
from sentence_transformers import SentenceTransformer

import os

try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Setup ---
client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma = chromadb.PersistentClient(path="./memory")
collection = chroma.get_or_create_collection(name="research_memory")

# --- Web Search ---
def web_search(query):
    print("  Searching the web...")
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": 3
        }
    )
    results = response.json().get("results", [])
    output = ""
    for r in results:
        output += f"Title: {r['title']}\n"
        output += f"URL: {r['url']}\n"
        output += f"Summary: {r['content']}\n\n"
    return output if output else "No results found."

# --- Save to Memory ---
def save_to_memory(question, answer):
    text = f"Q: {question}\nA: {answer}"
    embedding = embedder.encode(text).tolist()
    existing = collection.count()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[f"mem_{existing + 1}"]
    )
    print("  Saved to memory.")

# --- Check Memory ---
def check_memory(question):
    if collection.count() == 0:
        return None
    embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    top = results["documents"][0][0]
    distance = results["distances"][0][0]

    # If distance is low, memory is relevant
    if distance < 0.5:
        print("  Found relevant memory! Skipping web search.")
        return top
    return None

# --- Agent ---
def ask(question):
    print(f"\nQuestion: {question}")
    print("Thinking...\n")

    # Step 1: Check memory first
    memory = check_memory(question)

    if memory:
        # Use memory instead of searching
        context = f"You already researched this before. Here is what you found:\n{memory}"
        source = "memory"
    else:
        # Search the web
        context = web_search(question)
        source = "web"

    # Step 2: Generate answer
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant. Answer clearly based on the context provided."
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext ({source}):\n{context}"
            }
        ]
    )

    answer = response.choices[0].message.content

    # Step 3: Save to memory if we searched the web
    if source == "web":
        save_to_memory(question, answer)

    print(f"\nSource: {source.upper()}")
    print(f"Answer:\n{answer}")
    return answer

# --- Test ---
if __name__ == "__main__":
    # Run this twice to see memory in action!
    ask("What is quantum computing?")
    print("\n" + "="*50 + "\n")
    ask("Explain quantum computing to me")