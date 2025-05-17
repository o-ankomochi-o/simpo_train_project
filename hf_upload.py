import os

from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/home/www_kyoko_ogawa/simpo_train_project/output/generation-evaluation-trainer",
    repo_id="collabo-m/DiverseGen-KTO",
    repo_type="model",
)
