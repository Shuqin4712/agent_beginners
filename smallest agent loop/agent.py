import threading
import ollama
from tool_schemas import TOOLS_SCHEMA, TOOL_FUNCTIONS

MODEL = "qwen2.5:7b"

SYSTEM_PROMPT = """你是一个严格使用工具的助手。规则：
1. 所有数学计算必须调用 calculator 工具，不得自行计算
2. 所有知识性问题必须调用 search 工具，不得凭记忆回答
3. 工具返回错误时，如实告知用户，不要自行解释"""


def call_model_with_timeout(messages: list, timeout: int):
    """在独立线程中调用模型，支持真正的超时控制"""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = ollama.chat(model=MODEL, messages=messages, tools=TOOLS_SCHEMA)
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None, "模型响应超时，已终止"
    if error[0]:
        return None, f"调用模型出错: {error[0]}"
    return result[0], None


def run_agent(user_input: str, max_steps: int = 5, timeout_per_call: int = 30) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    print(f"\n🧑 用户输入: {user_input}\n{'=' * 50}")

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")

        response, err = call_model_with_timeout(messages, timeout=timeout_per_call)
        if err:
            return err

        msg = response.message
        messages.append(msg)

        if not msg.tool_calls:
            print("✅ 模型直接回答（无工具调用）")
            return msg.content

        print(f"🔧 模型决定调用 {len(msg.tool_calls)} 个工具:")
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"   → {name}({args})")

            func = TOOL_FUNCTIONS.get(name)
            try:
                result = func(**args) if func else f"未知工具: {name}"
            except Exception as e:
                result = f"工具执行出错: {e}"

            print(f"   ← 结果: {result}")
            messages.append({"role": "tool", "name": name, "content": str(result)})

    return "已达到最大步数，仍未得到最终答案"


if __name__ == "__main__":
    answer = run_agent("帮我算一下 (23 + 7) / 5，然后告诉我什么是 Ollama")
    print("\n最终答案:", answer)