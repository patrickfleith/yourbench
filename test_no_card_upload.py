#!/usr/bin/env python3
"""
Test script to verify that dataset card upload can be disabled.
"""

from datasets import Dataset
from yourbench.utils.dataset_engine import custom_save_dataset
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration with upload_card DISABLED
config = {
    "hf_configuration": {
        "hf_dataset_name": "test-no-card",
        "token": os.getenv("HF_TOKEN"),
        "private": True,
        "upload_card": False,  # This disables automatic card upload
    }
}

# Create test dataset
test_dataset = Dataset.from_dict({
    "text": ["sample 1", "sample 2"]
})

print("Testing disabled card upload...")
print(f"Upload card setting: {config['hf_configuration']['upload_card']}")

# This should NOT upload a card
custom_save_dataset(
    dataset=test_dataset,
    config=config,
    subset="test",
    push_to_hub=False,  # Don't actually push to avoid creating repos
    save_local=False
)

print("Completed - no card upload should have occurred.")
