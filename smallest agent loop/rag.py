import requests
import math
from ollama import chat


# =========================
# 1. chunk
# =========================

def chunk_text(text: str, **kwargs) -> list[str]:
    """按句子切分，每句话是一个完整的语义单元"""
    sentences = [s.strip() for s in text.split("。") if s.strip()]
    return sentences


# =========================
# 2. embedding
# =========================

def embed(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    return response.json()["embedding"]


# =========================
# 3. cosine similarity
# =========================

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2 + 1e-8)


# =========================
# 4. retrieve
# =========================

def retrieve(query: str, chunks: list, chunk_vectors: list, top_k: int = 2):

    query_vec = embed(query)

    scores = []
    for i, vec in enumerate(chunk_vectors):
        score = cosine_similarity(query_vec, vec)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    results = []
    for rank, (score, idx) in enumerate(scores[:top_k]):
        results.append({
            "chunk": chunks[idx],
            "score": score,
            "id": rank + 1   # citation编号
        })

    return results


# =========================
# 5. build_prompt（关键）
# =========================

def build_prompt(query: str, results: list):

    context = ""

    for r in results:
        context += f"[{r['id']}] {r['chunk']}\n"

    prompt = f"""
你是一个基于知识库回答问题的助手。

必须遵守：
- 只能使用提供的上下文回答
- 如果信息不足，说“不知道”
- 每个事实后必须标注来源编号 [1] [2]

上下文：
{context}

问题：
{query}

请开始回答：
"""

    return prompt


# =========================
# 6. answer_with_citations
# =========================

def answer_with_citations(query: str, chunks: list, chunk_vectors: list) -> str:

    # 1. retrieve
    results = retrieve(query, chunks, chunk_vectors, top_k=2)

    # 2. build prompt
    prompt = build_prompt(query, results)

    # 3. call ollama
    response = chat(
        model="qwen2.5:7b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# =========================
# 7. 测试数据
# =========================

test_text = """
Python是一种解释型、动态类型的编程语言，广泛用于数据科学、Web开发和自动化。
Agent是能感知环境、自主决策并执行动作的程序，是AI系统的核心概念。
Ollama是一个可以在本地运行大语言模型的工具，支持Llama、Qwen等多种模型。
LangChain是一个用于构建LLM应用的框架，提供了链式调用、记忆管理等功能。
ChromaDB是一个向量数据库，专门用于存储和检索embedding向量。
"""


# =========================
# 8. 建库
# =========================

chunks = chunk_text(test_text)

chunk_vectors = [embed(chunk) for chunk in chunks]