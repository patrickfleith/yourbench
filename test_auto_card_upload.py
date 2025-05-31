#!/usr/bin/env python3
"""
Test script to verify automatic dataset card upload via custom_save_dataset configuration.
"""

from datasets import Dataset
from yourbench.utils.dataset_engine import custom_save_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
import os

load_dotenv()

# Configuration with upload_card enabled
config = {
    "hf_configuration": {
        "hf_dataset_name": "patrickfleith/test-auto-card-upload",
        "token": os.getenv("HF_TOKEN"),
        "private": True,
        "upload_card": True,  # This enables automatic card upload
        "pretty_name": "Auto Card Upload Test",
        "language": "en",
        "license": "mit",
        "footer": "*(automatically uploaded via configuration)*"
    }
}

# Create test dataset
test_dataset = Dataset.from_dict({
    "question": ["What is AI?", "How does ML work?"],
    "answer": ["Artificial Intelligence...", "Machine Learning works by..."],
    "difficulty": ["easy", "medium"]
})

print("Testing automatic card upload via custom_save_dataset...")
print(f"Dataset: {len(test_dataset)} rows, {len(test_dataset.features)} columns")
print(f"Upload card enabled: {config['hf_configuration']['upload_card']}")

# Create repository first
api = HfApi()
token = os.getenv("HF_TOKEN")
repo_name = config["hf_configuration"]["hf_dataset_name"]

try:
    api.create_repo(
        repo_id=repo_name,
        repo_type="dataset", 
        private=True,
        token=token,
        exist_ok=True
    )
    print(f"Repository {repo_name} ready")
except Exception as e:
    print(f"Repository setup error: {e}")

# Test automatic card upload via custom_save_dataset
print("\nCalling custom_save_dataset with upload_card config enabled...")
custom_save_dataset(
    dataset=test_dataset,
    config=config, 
    subset="auto_test",
    push_to_hub=True,
    save_local=False
)

print("\nTest completed! Check the logs above for card upload.")
print(f"Visit: https://huggingface.co/datasets/{repo_name}")
