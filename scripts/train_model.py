import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import numpy as np
from typing import Optional, Dict, Any
from huggingface_hub import login
import os
from datasets import load_dataset, Dataset
import pandas as pd
from trl import SFTTrainer

import shutil

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

# # Clear /content (if running in Colab)
# content_path = "/content"
# if os.path.exists(content_path):
#     for item in os.listdir(content_path):
#         item_path = os.path.join(content_path, item)
#         if os.path.isdir(item_path):
#             shutil.rmtree(item_path)
#         else:
#             os.remove(item_path)
#     print("/content cleared!")



def main():
    # login to hugging face
    hug_token = os.getenv("HUGGINGFACE_TOKEN")
    print(hug_token)
    login(token=hug_token)
    #save_path = "/content/drive/MyDrive/models/SmolLM3-3B"


    # load model and tokenizer from huggingface
    instruct_model_name = "HuggingFaceTB/SmolLM3-3B"

    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_model_name, trust_remote_code=True)
    instruct_model = AutoModelForCausalLM.from_pretrained(
        instruct_model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("Model and tokenizer loaded!")

    # data
    ds = load_dataset("UNDP/sdgi-corpus")
    print("Dataset loaded!")

    train = pd.DataFrame(ds["train"])
    test = pd.DataFrame(ds["test"])
    print(f"Shape of train: {train.shape}")
    print(f"Shape of test: {test.shape}")
    print("Splits created!")

    sdg_labels = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well‑being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals"
    }

    def map_str_labels(row):
        final_labels = [sdg_labels[value] for value in row]
        return final_labels



    train["labels_str"] = train["labels"].apply(map_str_labels)
    test["labels_str"] = test["labels"].apply(map_str_labels)

    # extract of the text
    train['language'] = train['metadata'].apply(lambda x: x['language'])
    test['language'] = test['metadata'].apply(lambda x: x['language'])

    # get unique languages
    print(np.unique(train["language"]))


    # extract language-specific for tuning + tsting
    train_en = train[train["language"] == "en"]
    test_en =  test[test["language"]=="en"]
    print(f"Size of train in english: {train_en.shape}")
    print(f"Size of test in english: {test_en.shape}")
    print("-------------------------------")

    # extract for testing
    test_es = pd.concat([test[test["language"] == "es"], train[train["language"] == "es"]],axis=0)
    test_fr = pd.concat([test[test["language"] == "fr"], train[train["language"] == "fr"]],axis=0)
    print(f"Size of test in french: {test_fr.shape}")
    print(f"Size of test in spanish: {test_es.shape}")

    print("Language-specific splits created!")


    def format_for_chat(example):
        gold_labels = example["labels"]
        gold_json = str(gold_labels)

        example["messages"] = [
            {"role": "system", "content": "You are a classifier that assigns SDG labels to text."},

            {"role": "user", "content":
                f"Text: {example['text']}\n\n"
                f"Available labels: {list(sdg_labels.values())}\n"
                "Return only the relevant labels as a JSON list. Skip returning any other text."
            },

            {"role": "assistant", "content": gold_json}
        ]
        return example

    train_chat = train_en.apply(format_for_chat, axis=1)


    #format for model training
    def format_chat(example):
        # convert messages → training string
        example["text"] = instruct_tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False, #SFT trainer is doing it later on
            add_generation_prompt=False  # we don't include assistant output during training
        )
        return example


    # convert for training
    train_dataset = train_chat.apply(format_chat,axis=1)

    # get only relventat col for training
    formatted_dataset = train_dataset.drop([col for col in train_dataset.columns if col != "messages"],axis=1)


    def train_model():
        from peft import LoraConfig, TaskType

        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            lora_dropout=0.15,
            bias="none", #for memory efficency
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],  # define trainable layers for SmolLM2
        )

        from trl import SFTTrainer, SFTConfig

        # Configure training parameters
        training_config = SFTConfig(
        # Model and data
        output_dir=f"./{instruct_model_name}",
        dataset_text_field="text",
        max_length=4096,

        # Training hyperparameters
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=32,
        learning_rate=5e-5,
        num_train_epochs=1, 
        #max_steps=1,  # Limit steps for demo

        # Optimization
        warmup_steps=10,
        weight_decay=0.01,
        optim="adamw_torch",

        # Logging and saving
        logging_steps=10,
        save_steps=10,
        eval_steps=100,
        save_total_limit=2,

        # Memory optimization
        dataloader_num_workers=1,
        group_by_length=True,  # Group similar length sequences

        fp16=True,
        bf16=False,

        # Hugging Face Hub integration
        push_to_hub=False,  # Set to True to upload to Hub
        hub_model_id= f"your-username/{instruct_model_name}",

        # Experiment tracking
        report_to= "none",#["trackio"],  # Use trackio for experiment tracking
        run_name=f"{instruct_model_name}-training",
        )


        print("Training configuration set!")
        print(f"Effective batch size: {training_config.per_device_train_batch_size * training_config.gradient_accumulation_steps}")


        dataset2 = Dataset.from_pandas(formatted_dataset)  # Converts pd.DataFrame -> HF Dataset



        lora_trainer = SFTTrainer(
            model=instruct_model,
            train_dataset=dataset2,
            peft_config=lora_config,  # if using LoRA
            #max_seq_length=training_config.max_length,
            args=training_config,
        )


        print("Starting LoRA training…")
        lora_trainer.train()

        MODEL_SAVE_PATH ="models"
        modelNumber = 1  # replace with your numbering logic

        hf_save_path = os.path.join(MODEL_SAVE_PATH, f'{modelNumber}_smollm3_4096')
        os.makedirs(hf_save_path, exist_ok=True)
        # Save the model & tokenizer
        lora_trainer.model.save_pretrained(hf_save_path)
        lora_trainer.tokenizer.save_pretrained(hf_save_path)

        print(f"SmolLM3 model and tokenizer saved to {hf_save_path}")

    train_model()



if __name__ == "__main__":
    main()
