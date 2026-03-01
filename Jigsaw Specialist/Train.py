import torch
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
JIGSAW_CSV = "/content/train.csv"
GOLDEN_JSON = "/content/golden_test_set.json"
OUTPUT_DIR = "qwen2.5-7b-jigsaw-lora"
MAX_LENGTH = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

with open(GOLDEN_JSON, "r") as f:
    gold_data = json.load(f)
test_ids = [item['row_id'] for item in gold_data['jigsaw']]

df = pd.read_csv(JIGSAW_CSV)
train_df = df[~df['row_id'].isin(test_ids)].copy()

def tokenize_function(examples):
    system_prompt = "You are a forum moderator. Analyze if the comment violates the specific rule. Answer with only 'yes' or 'no'."
    # Construct the ChatML content
    texts = []
    for i in range(len(examples['body'])):
        body = examples['body'][i]
        rule = examples['rule'][i]
        violation = "yes" if examples['rule_violation'][i] == 1 else "no"
        full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nRule: {rule}\nComment: {body}\nDoes this comment violate the rule? Answer Yes or No.<|im_end|>\n<|im_start|>assistant\n{violation}<|im_end|>"
        texts.append(full_text)

    
    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)

raw_dataset = Dataset.from_pandas(train_df)
train_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=raw_dataset.column_names)

bnb_config = BitsAndBytesConfig(
    load_in_4_bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

###########################################
# LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

###########################################
# Training parameters
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    bf16=True,
    save_strategy="no",
    optim="paged_adamw_32bit",
    report_to="none",
    remove_unused_columns=False 
)

###########################################
# Training
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

trainer.model.save_pretrained("jigsaw_adapter")
print("Training Finished, Adapter saved")
