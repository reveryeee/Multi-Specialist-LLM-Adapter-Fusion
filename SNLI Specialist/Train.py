import torch
import json
import pandas as pd
from datasets import load_dataset, Dataset
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
GOLDEN_JSON = "/content/golden_test_set.json"
OUTPUT_DIR = "qwen2.5-7b-snli-lora"
MAX_SAMPLES = 10000  # extract 10000 rows
MAX_LENGTH = 512     # SNLI has short sentences

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token


with open(GOLDEN_JSON, "r") as f:
    gold_data = json.load(f)

gold_pairs = set([f"{item['premise']}_{item['hypothesis']}" for item in gold_data['snli']])


raw_snli = load_dataset("snli", split="train", trust_remote_code=True)
snli_labels = {0: "entailment", 1: "neutral", 2: "contradiction"}

def tokenize_function(examples):
    system_prompt = "Answer with only one word: entailment, neutral, or contradiction."
    texts = []
    for i in range(len(examples['premise'])):
        p = examples['premise'][i]
        h = examples['hypothesis'][i]
        label_id = examples['label'][i]

       
        if label_id not in snli_labels or f"{p}_{h}" in gold_pairs:
            continue

        violation = snli_labels[label_id]
        full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nP: {p}\nH: {h}<|im_end|>\n<|im_start|>assistant\n{violation}<|im_end|>"
        texts.append(full_text)

    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)

train_dataset = raw_snli.shuffle(seed=42).select(range(MAX_SAMPLES + 500)) 
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=raw_snli.column_names)
train_dataset = train_dataset.select(range(MAX_SAMPLES)) 

# ==========================================
# Load model and LoRA

bnb_config = BitsAndBytesConfig(
    load_in_4_bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ==========================================
# Training parameters

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8, # SNLI has short sequences，Batch could be a bit bigger
    gradient_accumulation_steps=2,
    learning_rate=1e-4,            # Low learning rate is more stable
    logging_steps=20,
    num_train_epochs=2,            # There are 10000 rows of data, 2 epoches are big enough
    bf16=True,
    save_strategy="no",
    optim="paged_adamw_32bit",
    report_to="none",
    remove_unused_columns=False
)

# ==========================================
# Training
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

trainer.model.save_pretrained("snli_adapter")
print("Training finished")
