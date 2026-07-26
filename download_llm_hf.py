import os
import shutil
import time
from huggingface_hub import hf_hub_download

repo_id = "bartowski/Qwen2.5-14B-Instruct-GGUF"
filename = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
dest_dir = os.path.join("models", "llm")
final_path = os.path.join(dest_dir, "model.gguf")

os.makedirs(dest_dir, exist_ok=True)

success = False
while not success:
    try:
        print("Attempting to download/resume...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        success = True
    except Exception as e:
        print(f"Connection dropped: {e}")
        print("Retrying in 2 seconds...")
        time.sleep(2)

print(f"Downloaded to {downloaded_path}. Renaming to {final_path}...")
if os.path.exists(final_path):
    os.remove(final_path)
os.rename(downloaded_path, final_path)
print("Download and rename complete!")
