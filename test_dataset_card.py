#!/usr/bin/env python3
"""
Simple test script to demonstrate dataset card upload functionality.
Run this script to test the dataset card upload feature.
"""

from datasets import Dataset
from yourbench.utils.dataset_engine import upload_dataset_card
from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

load_dotenv()

# Create a simple test configuration
config = {
    "hf_configuration": {
        "hf_dataset_name": "test-dataset-card",
        "hf_organization": None,  # Will use your username
        "token": os.getenv("HF_TOKEN"),  # Include token in config
        "private": True,
        "pretty_name": "Test Dataset Card",
        "language": "en",
        "license": "apache-2.0",
        "footer": "*(This is a test dataset card)*"
    }
}

# Create a simple test dataset
test_dataset = Dataset.from_dict({
    "id": [1, 2, 3],
    "text": ["Hello", "World", "Test"],
    "label": ["A", "B", "C"]
})

print("Testing dataset card upload...")
print("Make sure you have HF_TOKEN set in your environment!")
print(f"Dataset: {len(test_dataset)} rows, {len(test_dataset.features)} columns")

# First create the repository and upload the dataset
api = HfApi()
token = os.getenv("HF_TOKEN")
username = api.whoami(token)["name"]
repo_name = f"{username}/{config['hf_configuration']['hf_dataset_name']}"

try:
    print(f"Creating repository: {repo_name}")
    api.create_repo(
        repo_id=repo_name,
        repo_type="dataset",
        private=True,
        token=token,
        exist_ok=True  # Don't fail if repo already exists
    )
    print("Repository created/verified")
    
    # Upload the dataset first
    print("Uploading test dataset...")
    test_dataset.push_to_hub(repo_name, private=True, token=token)
    print("Dataset uploaded successfully")
    
    # Update config with full repo name
    config["hf_configuration"]["hf_dataset_name"] = repo_name
    
except Exception as e:
    print(f"Setup error: {e}")

# Test the upload function with config-based approach
print("Testing automatic dataset card upload via configuration...")
upload_dataset_card(config, test_dataset)

print("Test completed. Check the logs above for any issues.")
