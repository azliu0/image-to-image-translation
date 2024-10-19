"""Load weights from HuggingFace to a modal volume."""

import modal

app = modal.App("load-hf-weights")
image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub")
volume = modal.Volume.from_name("image-to-image-translation-weights", create_if_missing=True)

@app.function(volumes={"/root/data": volume}, image=image)
def load_weights():
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    
    repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    files = [
        ("v1-5-pruned-emaonly.ckpt", "/root/data/model.ckpt"),
        ("tokenizer/vocab.json", "/root/data/tokenizer_vocab.json"),
        ("tokenizer/merges.txt", "/root/data/tokenizer_merges.txt")
    ]
    
    for file, destination in files:
        local_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir="/root/data")
        Path(local_path).rename(destination)
        print(f"Downloaded {file} to {destination}")

    volume.commit()

if __name__ == "__main__":
    with modal.enable_output():
        with app.run():
            load_weights.remote()

