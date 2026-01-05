from huggingface_hub import snapshot_download

try:
    # 这不会重新下载，只会返回已下载文件的路径
    path = snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
    print("\n✅ 模型权重存放在这里:")
    print(path)
except Exception as e:
    print("❌ 尚未下载该模型，或无法连接 HuggingFace")