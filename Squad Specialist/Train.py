import torch
import json
from datasets import load_dataset
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
OUTPUT_DIR = "qwen2.5-7b-squad-lora"
MAX_SAMPLES = 8000   # 8000 rows of high-quality data for reading comprehension
MAX_LENGTH = 1024    # 1024 length include the article background


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

with open(GOLDEN_JSON, "r") as f:
    gold_data = json.load(f)
test_ids = set([item['id'] for item in gold_data['squad']])

raw_squad = load_dataset("squad_v2", split="train")

def tokenize_function(examples):
    system_prompt = "Answer the question based on context. Shortest answer possible. If no answer, say 'unanswerable'."
    texts = []
    for i in range(len(examples['id'])):
    
        if examples['id'][i] in test_ids:
            continue

        ctx = examples['context'][i]
        q = examples['question'][i]

        ans_list = examples['answers'][i]['text']
        answer = ans_list[0] if ans_list else "unanswerable"

        full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nC: {ctx}\nQ: {q}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        texts.append(full_text)

    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)

train_dataset = raw_squad.shuffle(seed=42).select(range(MAX_SAMPLES + 500))
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=raw_squad.column_names)
train_dataset = train_dataset.select(range(MAX_SAMPLES))

# ==========================================
# Load Model and LoRA

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
    per_device_train_batch_size=4,  # Batch size = 4 is stable when the length is 1024
    gradient_accumulation_steps=4,  # updates when the gradient accumulates to 16
    learning_rate=1e-4,
    logging_steps=10,
    num_train_epochs=2,
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

trainer.model.save_pretrained("squad_adapter")
print("Training finished")
