from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="TourismPackagePredictionAnalysis/deployment",     # the local folder containing your files
    repo_id="sumitsinha2603/TourismPackagePredictionAnalysisSpace",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
