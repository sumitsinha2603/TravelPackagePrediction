from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo, upload_file
import os

repo_id = "sumitsinha2603/TourismPackagePredictionAnalysis"
repo_type = "dataset"

# Initialize API client
HF_TOKEN = os.environ('hf_token')
api = HfApi(token=HF_TOKEN)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    path_or_fileobj=dataset_file,
    path_in_repo="train.csv",
    repo_id=repo_id,
    repo_type="dataset"
)
