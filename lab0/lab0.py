"""
Lab0 - 環境檢查
================
確認 vLLM API、Docker、Embedding Model 三項環境就緒。
"""

import os
import sys

# ==============================================================================
#                          vLLM 設定（與前面課程相同）
# ==============================================================================

ENDPOINT = "http://localhost:8299/v1"
MODELNAME = "Qwen2.5-3B-Instruct"


def test_vllm():
    """測試 vLLM API 是否可連線並正常回應"""
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=ENDPOINT)

    messages = [
        {"role": "system", "content": "你是一個測試助理。"},
        {"role": "user", "content": "請用一句話介紹自己。"},
    ]

    print(f"[vLLM] 端點: {ENDPOINT}")
    print(f"[vLLM] 模型: {MODELNAME}")

    response = client.chat.completions.create(
        model=MODELNAME,
        messages=messages,
        temperature=0.7,
        max_tokens=128,
    )
    reply = response.choices[0].message.content
    print(f"[vLLM] 回應: {reply}")
    print("[vLLM] ✓ 連線成功\n")
    return True


# ==============================================================================
#                          Docker 環境測試
# ==============================================================================

def test_docker():
    """測試 Docker 是否已安裝"""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10
        )
        print(f"[Docker] {result.stdout.strip()}")
        print("[Docker] ✓ Docker 已安裝\n")
        return True
    except FileNotFoundError:
        print("[Docker] ✗ Docker 未安裝，請先安裝 Docker Desktop 或等效工具。")
        return False
    except Exception as e:
        print(f"[Docker] ✗ 測試失敗: {e}")
        return False


# ==============================================================================
#                          Embedding Model 測試
# ==============================================================================

def test_embedding():
    """測試 sentence-transformers embedding model 是否可載入"""
    from langchain.embeddings import HuggingFaceEmbeddings

    model_name = "intfloat/multilingual-e5-large"
    print(f"[Embedding] 載入模型: {model_name}（首次下載可能需要幾分鐘）...")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )

    test_text = "這是一段測試文字，用來確認 embedding 模型可以正常運作。"
    vector = embeddings.embed_query(test_text)
    print(f"[Embedding] 測試文字: {test_text}")
    print(f"[Embedding] 向量維度: {len(vector)}")
    print("[Embedding] ✓ 模型載入成功\n")
    return True


# ==============================================================================
#                          主程式
# ==============================================================================

if __name__ == "__main__":
    results = {}

    print("=" * 60)
    print("Lab0：環境檢查")
    print("=" * 60 + "\n")

    # 1. vLLM
    try:
        results["vLLM"] = test_vllm()
    except Exception as e:
        print(f"[vLLM] ✗ 連線失敗: {e}\n")
        results["vLLM"] = False

    # 2. Docker
    results["Docker"] = test_docker()

    # 3. Embedding
    try:
        results["Embedding"] = test_embedding()
    except Exception as e:
        print(f"[Embedding] ✗ 載入失敗: {e}\n")
        results["Embedding"] = False

    # 總結
    print("=" * 60)
    print("環境檢查結果：")
    for name, ok in results.items():
        status = "✓ 通過" if ok else "✗ 未通過"
        print(f"  {name}: {status}")
    print("=" * 60)

    if all(results.values()):
        print("\n所有環境檢查通過，可以開始 Lab1！")
    else:
        print("\n部分環境未通過，請依上方錯誤訊息排除後重試。")
