import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import PeftModel, PeftConfig

import os
import shutil

language = 'en'


# Clear Hugging Face cache
hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
    print("Hugging Face cache cleared!")

# Clear PyTorch cache
torch_cache = os.path.expanduser("~/.cache/torch")
if os.path.exists(torch_cache):
    shutil.rmtree(torch_cache)
    print("PyTorch cache cleared!")

# 1. Setup
sdg_labels = {
    1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Wellbeing",
    4: "Quality Education", 5: "Gender Equality", 6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy", 8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure", 10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities", 12: "Responsible Consumption and Production",
    13: "Climate Action", 14: "Life Below Water", 15: "Life on Land",
    16: "Peace, Justice and Strong Institutions", 17: "Partnerships for the Goals"
}

SAVE_PATH = f'./finetuned_predictions_{language}.json'

model_id = "HuggingFaceTB/SmolLM3-3B"

lora_id = "miry-itu/smol-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, device_map="auto", padding_side='left')

# 2. Define Custom Collate Function (The Fix)
def custom_collate_fn(batch):
    # 'batch' is a list of dictionaries (one per item in the batch)

    # 1. Extract inputs for the model and pad them
    # We use tokenizer.pad to handle input_ids and attention_mask padding automatically
    model_inputs = tokenizer.pad(
        [{"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]} for x in batch],
        padding=True,
        return_tensors="pt"
    )

    # 2. Extract metadata (keep as simple Python lists)
    file_ids = [x["file_id"] for x in batch]
    gold_labels = [x["gold_labels"] for x in batch]

    # 3. Combine them
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "file_id": file_ids,
        "gold_labels": gold_labels
    }

# 3. Processing Function
def format_for_chat(example):
    file_id = example['metadata']['file_id']
    messages = [
        {"role": "system", "content": "You are a specialized classifier. You output ONLY valid JSON arrays of SDG labels."},
        {"role": "user", "content": (
            f"Text: {example['text']}\n"
            f"Labels: {list(sdg_labels.values())}\n"
            "Constraint: Return a JSON array of the relevant SDG labels."
        )}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Return standard lists (not tensors yet)
    tokenized = tokenizer(prompt, truncation=True, max_length=4092) # Increased length slightly
    tokenized['file_id'] = file_id
    tokenized['gold_labels'] = example['labels']
    return tokenized

# 4. Load & Prepare Data
base_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
model = PeftModel.from_pretrained(base_model, lora_id)

ds = load_dataset("UNDP/sdgi-corpus", split="test")
ds = ds.filter(lambda x: x['metadata']['language'] == language)

# Map the data
tokenized_dataset = ds.map(format_for_chat, batched=False)

# IMPORTANT: Do NOT use set_format(type='torch') or output_all_columns=True.
# We let the custom_collate_fn handle the conversion.

test_dataloader = DataLoader(
    tokenized_dataset,
    batch_size=1,
    collate_fn=custom_collate_fn # Use our custom function
)

# 5. Inference Loop
model.eval()
all_results = []

with torch.no_grad():
    for batch in test_dataloader:
        # Move ONLY inputs to CUDA
        inputs = {
            "input_ids": batch["input_ids"].to("cuda"),
            "attention_mask": batch["attention_mask"].to("cuda")
        }

        # Metadata stays on CPU
        current_ids = batch['file_id']
        current_golds = batch['gold_labels']

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )

        input_len = inputs["input_ids"].shape[1]
        decoded_preds = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for i in range(len(decoded_preds)):
            all_results.append({
                "file_id": current_ids[i],
                "gold": current_golds[i], # It's already a list, no need for .tolist()
                "prediction_text": decoded_preds[i]
            })

        if len(all_results) % 10 == 0:
            with open(SAVE_PATH, 'w') as f:
                json.dump(all_results, f, indent=4)

        # Clean up
        del inputs, outputs

# Final save
with open(SAVE_PATH, 'w') as f:
    json.dump(all_results, f, indent=4)