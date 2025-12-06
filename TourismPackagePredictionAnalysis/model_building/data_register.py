from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo, upload_file
import os

# HF token from GitHub Actions environment
HF_TOKEN = os.environ["HF_TOKEN"]

repo_id = "sumitsinha2603/TourismPackagePredictionAnalysis"
repo_type = "dataset"
dataset_file = "train.csv"   # path to your file

api = HfApi(token=HF_TOKEN)

# Step 1: Check if repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print("Dataset repo created.")

# Step 2: Upload file
upload_file(
    path_or_fileobj=dataset_file,
    path_in_repo="train.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=HF_TOKEN
)

print("File uploaded successfully.")
