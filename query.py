import requests
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
BASE_URL = os.getenv("PROXY_BASE_URL")
DB_PATH = "./faiss_index"

# Load embeddings + FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)


# --- RETRIEVE ---
def retrieve(query):
    docs = db.similarity_search(query, k=3)
    context = ""
    for d in docs:
        if d.page_content:
            context += f"{d.page_content}\n(Source: {d.metadata['source']}, Page {d.metadata['page']})\n\n"
    return context


# --- LLM CALL USING REQUESTS ---
def call_llm(system_prompt, user_prompt):
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "openai/gpt-4.1-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        },
    )

    data = response.json()
    return data["choices"][0]["message"]["content"]


# --- AGENTS ---
methods_prompt = """You are a Methods Analyst.
Focus on methods, datasets, and setup.
Always cite (Source: filename, Page X)."""

results_prompt = """You are a Results Extractor.
Focus on numerical results and comparisons.
Always cite (Source: filename, Page X)."""

review_prompt = """You are a Skeptical Reviewer.
Focus on limitations and critique.
Always cite (Source: filename, Page X)."""

general_prompt = """You are a General Synthesizer.
Answer clearly and concisely.
Always cite (Source: filename, Page X)."""

router_prompt = """You are a router.

Return ONLY one word:
methods, results, review, or general.
"""


# --- ROUTER ---
def route(query):
    decision = call_llm(router_prompt, query).lower()

    if "method" in decision:
        return "methods"
    elif "result" in decision:
        return "results"
    elif "review" in decision or "limit" in decision:
        return "review"
    else:
        return "general"


# --- MAIN LOOP ---
def main():
    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        context = retrieve(query)

        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

        route_decision = route(query)

        if route_decision == "methods":
            answer = call_llm(methods_prompt, user_prompt)
            print("\n[Methods Analyst]\n", answer)

        elif route_decision == "results":
            answer = call_llm(results_prompt, user_prompt)
            print("\n[Results Extractor]\n", answer)

        elif route_decision == "review":
            answer = call_llm(review_prompt, user_prompt)
            print("\n[Skeptical Reviewer]\n", answer)

        else:
            answer = call_llm(general_prompt, user_prompt)
            print("\n[General Synthesizer]\n", answer)


if __name__ == "__main__":
    main()