import torch
import pandas as pd
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
JIGSAW_PATH = "train.csv" #Only Jigsaw dataset was uploaded on Colab, for SNLI and Squad_v2, we read them via hugging face
SAMPLE_SIZE = 500

# Preparing the Golden Test Set)
golden_test_set = {"snli": [], "jigsaw": [], "squad": []}

# SNLI
snli_labels = {0: "entailment", 1: "neutral", 2: "contradiction"}

# Add `shuffle(seed=42)` to ensure randomness, and use an iterator to filter out invalid data with label -1
snli_iter = iter(load_dataset("snli", split="validation", trust_remote_code=True).shuffle(seed=42))
while len(golden_test_set["snli"]) < SAMPLE_SIZE:
    item = next(snli_iter)
    if item['label'] in snli_labels:
        golden_test_set["snli"].append({
            "premise": item['premise'],
            "hypothesis": item['hypothesis'],
            "label_id": item['label'],
            "label_text": snli_labels[item['label']]
        })

# Jigsaw Agile 
jigsaw_df = pd.read_csv(JIGSAW_PATH)
# 500 rows randomly
jigsaw_test_df = jigsaw_df.sample(n=SAMPLE_SIZE, random_state=42)
for _, row in jigsaw_test_df.iterrows():
    golden_test_set["jigsaw"].append({
        "row_id": int(row['row_id']),
        "body": str(row['body']),
        "rule": str(row['rule']),
        "rule_violation": int(row['rule_violation'])
    })

# SQuAD v2 
# Extract the top 500 sentences after direct shuffling.
squad_test = load_dataset("squad_v2", split="validation").shuffle(seed=42).select(range(SAMPLE_SIZE))
for item in squad_test:
    golden_test_set["squad"].append({
        "id": item['id'],
        "context": item['context'],
        "question": item['question'],
        "answers": item['answers']['text'] 
    })

# store the golden test set locally, this is for testing for all models
with open("golden_test_set.json", "w", encoding="utf-8") as f:
    json.dump(golden_test_set, f, ensure_ascii=False, indent=4)


# Load the original Qwen2.5-7B-Instruct for testing

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)

def get_response(system_prompt, user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()

# Concuct the baseline testing

results = {"snli": 0, "jigsaw": 0, "squad": 0}

# Testing on SNLI
print("Testing SNLI...")
for item in tqdm(golden_test_set["snli"]):
    res = get_response(
        "Answer with only one word: entailment, neutral, or contradiction.",
        f"P: {item['premise']}\nH: {item['hypothesis']}"
    )
    if item['label_text'] in res:
        results["snli"] += 1

# Testing on Jigsaw
print("Testing Jigsaw Agile...")
for item in tqdm(golden_test_set["jigsaw"]):
    user_input = f"Rule: {item['rule']}\nComment: {item['body']}\nDoes this comment violate the rule? Answer Yes or No."
    res = get_response(
        "You are a forum moderator. Analyze if the comment violates the specific rule. Answer with only 'yes' or 'no'.",
        user_input
    )
    true_label = "yes" if item['rule_violation'] == 1 else "no"
    if true_label in res:
        results["jigsaw"] += 1

# Testing on Squad_v2
print("Testing SQuAD v2...")
for item in tqdm(golden_test_set["squad"]):
    res = get_response(
        "Answer the question based on context. Shortest answer possible. If no answer, say 'unanswerable'.",
        f"C: {item['context']}\nQ: {item['question']}"
    )
    answers = [str(a).lower() for a in item['answers']]
    if not answers: # in some rows, there are no answers
        if "unanswerable" in res:
            results["squad"] += 1
    else: 
        if any(a in res for a in answers):
            results["squad"] += 1


# Final testing report
print("\n" + "="*40)
print("[Baseline Results]")
print(f"SNLI Accuracy:   {results['snli']/SAMPLE_SIZE:.2%}")
print(f"Jigsaw Accuracy: {results['jigsaw']/SAMPLE_SIZE:.2%}")
print(f"SQuAD Match Rate:{results['squad']/SAMPLE_SIZE:.2%}")
print("="*40)
