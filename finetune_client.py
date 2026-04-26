#!/usr/bin/env python3
import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Step 1: Upload training file
print("Uploading train.jsonl...")
upload = client.files.create(
    file=open("train.jsonl", "rb"),
    purpose="fine-tune",
)
print(f"File uploaded: {upload.id}")

# Step 2: Create fine-tuning job
print("Starting fine-tuning job...")
job = client.fine_tuning.jobs.create(
    training_file=upload.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3,
    },
)
print(f"Job created: {job.id}")

# Step 3: Poll until done
while True:
    status = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {status.status}")

    if status.status == "succeeded":
        print(f"\nFine-tuning complete!")
        print(f"New model: {status.fine_tuned_model}")
        break
    elif status.status == "failed":
        print(f"\nFine-tuning failed: {status.error}")
        break
    elif status.status == "cancelled":
        print(f"\nFine-tuning cancelled")
        break

    time.sleep(30)
