# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
HF_TOKEN = userdata.get('hf_token')
api = HfApi(token=HF_TOKEN)
DATASET_PATH = "hf://datasets/sumitsinha2603/TourismPackagePredictionAnalysis/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
df.drop(columns=['CustomerID'], inplace=True)

# Remove any unnamed index-like columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Remove fully empty columns
df = df.dropna(axis=1, how="all")

# Encode categorical columns
label_encoder = LabelEncoder()
for col in ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']:
    df[col] = label_encoder.fit_transform(df[col])

# For Tourism Package Prediction dataset, target = ProdTaken (0/1)
target_column = "ProdTaken"

# Split data into features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="sumitsinha2603/TourismPackagePredictionAnalysis",
        repo_type="dataset",
    )
