from rag import chunk_text, embed, retrieve


def calculator(expression: str) -> str:
    """计算一个只包含数字和 + - * / ( ) 的数学表达式"""
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不允许的字符"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算出错: {e}"


def read_file(path: str) -> str:
    """读取本地文本文件内容（最多 2000 字符）"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()[:2000]
    except Exception as e:
        return f"读取文件出错: {e}"


_raw_text = """
Python是一种解释型、动态类型的编程语言，广泛用于数据科学、Web开发和自动化。
Agent是能感知环境、自主决策并执行动作的程序，是AI系统的核心概念。
Ollama是一个可以在本地运行大语言模型的工具，支持Llama、Qwen等多种模型。
LangChain是一个用于构建LLM应用的框架，提供了链式调用、记忆管理等功能。
ChromaDB是一个向量数据库，专门用于存储和检索embedding向量。
"""

_chunks = chunk_text(_raw_text)
_chunk_vectors = [embed(chunk) for chunk in _chunks]


def search(query: str) -> str:
    """在本地知识库中按向量检索返回相关片段"""
    results = retrieve(query, _chunks, _chunk_vectors, top_k=2)
    if not results:
        return "知识库中没有找到相关内容"
    return "\n".join(f"[{r['id']}] {r['chunk']}" for r in results)