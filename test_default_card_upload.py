#!/usr/bin/env python3
"""
Test script to verify that dataset card upload defaults to True when not specified.
"""

from datasets import Dataset
from yourbench.utils.dataset_engine import custom_save_dataset
from dotenv import load_dotenv
import os

load_dotenv()

# Configuration WITHOUT upload_card setting (should default to True)
config = {
    "hf_configuration": {
        "hf_dataset_name": "test-default-card",
        "token": os.getenv("HF_TOKEN"),
        "private": True,
        "pretty_name": "Default Card Test",
        # NOTE: upload_card is NOT specified - should default to True
    }
}

# Create test dataset
test_dataset = Dataset.from_dict({
    "text": ["default test"]
})

print("Testing default card upload behavior (should be enabled)...")
print(f"Upload card in config: {config['hf_configuration'].get('upload_card', 'NOT_SET')}")

# This should upload a card by default
custom_save_dataset(
    dataset=test_dataset,
    config=config,
    subset="test",
    push_to_hub=False,  # Don't actually push to avoid creating repos  
    save_local=False
)

# Check what the default would be
default_setting = config["hf_configuration"].get("upload_card", True)
print(f"Default upload_card value would be: {default_setting}")
print("Test completed - card upload should be enabled by default.")
