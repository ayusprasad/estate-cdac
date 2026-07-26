import urllib.request
import sys
import os

url = "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
dest = os.path.join("models", "llm", "model.gguf")

os.makedirs(os.path.dirname(dest), exist_ok=True)

def reporthook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading Qwen2.5-14B... {percent}%")
    sys.stdout.flush()

print(f"Downloading {url} to {dest}...")
urllib.request.urlretrieve(url, dest, reporthook)
print("\nDownload complete.")
