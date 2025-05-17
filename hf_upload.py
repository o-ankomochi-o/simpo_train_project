from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/path/to/local/model",
    repo_id="collabo-m/DiverseGen-KTO",
    repo_type="model",
)
