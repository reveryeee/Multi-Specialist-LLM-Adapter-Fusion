import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/content/snli_adapter"  
GOLDEN_JSON = "/content/golden_test_set.json"

# ==========================================
# Load fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ==========================================
# Load the test data
with open(GOLDEN_JSON, "r") as f:
    golden_test_set = json.load(f)

def get_response(system_prompt, user_input):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  
        )

    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()

# ==========================================
# Test on 3 scenarios

results = {"snli": 0, "jigsaw": 0, "squad": 0}
SAMPLE_SIZE = 500

# SNLI
print("Testing SNLI Specialist Strength...")
for item in tqdm(golden_test_set["snli"]):
    res = get_response(
        "Answer with only one word: entailment, neutral, or contradiction.",
        f"P: {item['premise']}\nH: {item['hypothesis']}"
    )
    if item['label_text'] in res:
        results["snli"] += 1

# Jigsaw
print("Testing Jigsaw Generalization...")
for item in tqdm(golden_test_set["jigsaw"]):
    user_input = f"Rule: {item['rule']}\nComment: {item['body']}\nDoes this comment violate the rule? Answer Yes or No."
    res = get_response(
        "You are a forum moderator. Analyze if the comment violates the specific rule. Answer with only 'yes' or 'no'.",
        user_input
    )
    true_label = "yes" if item['rule_violation'] == 1 else "no"
    if true_label in res:
        results["jigsaw"] += 1

# Squad_v2
print("Testing SQuAD v2 Generalization...")
for item in tqdm(golden_test_set["squad"]):
    res = get_response(
        "Answer the question based on context. Shortest answer possible. If no answer, say 'unanswerable'.",
        f"C: {item['context']}\nQ: {item['question']}"
    )
    answers = [str(a).lower() for a in item['answers']]
    if not answers:
        if "unanswerable" in res:
            results["squad"] += 1
    else:
        if any(a in res for a in answers):
            results["squad"] += 1

# ==========================================
# Testing report
print("\n" + "="*40)
print("[SNLI-Specialist Model Results]")
print(f"SNLI Score (Logic):     {results['snli']/SAMPLE_SIZE:.2%}")
print(f"Jigsaw Score (Safety):  {results['jigsaw']/SAMPLE_SIZE:.2%}")
print(f"SQuAD Score (Reading):  {results['squad']/SAMPLE_SIZE:.2%}")
print("="*40)
