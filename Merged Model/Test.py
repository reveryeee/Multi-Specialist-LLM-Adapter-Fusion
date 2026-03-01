import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GOLDEN_JSON = "/content/golden_test_set.json"

# We merged 3 specialist models together
ADAPTERS = {
    "jigsaw": "/content/jigsaw_adapter",
    "snli": "/content/snli_adapter",
    "squad": "/content/squad_adapter"
}

# ==========================================
# Load the base model

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# ==========================================
# Load the adapters
model = PeftModel.from_pretrained(base_model, ADAPTERS["jigsaw"], adapter_name="jigsaw")

model.load_adapter(ADAPTERS["snli"], adapter_name="snli")
model.load_adapter(ADAPTERS["squad"], adapter_name="squad")

# we use the add_weighted_adapter techniques and we define the weights of 3 models to 1:1:1 for now
model.add_weighted_adapter(
    adapters=["jigsaw", "snli", "squad"],
    weights=[1.0, 1.0, 1.0],
    adapter_name="merged_expert",
    combination_type="linear" # Linear weighted merge
)

# activate the merged adapters
model.set_adapter("merged_expert")
model.eval()

# ==========================================

with open(GOLDEN_JSON, "r") as f:
    golden_test_set = json.load(f)

def get_response(system_prompt, user_input, max_tokens=20):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_tokens, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False 
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().lower()

results = {"snli": 0, "jigsaw": 0, "squad": 0}
SAMPLE_SIZE = 500

## test on 3 scenarios

# SNLI
print("Testing Logic (SNLI)...")
for item in tqdm(golden_test_set["snli"]):
    res = get_response("Answer with only one word: entailment, neutral, or contradiction.", f"P: {item['premise']}\nH: {item['hypothesis']}")
    if item['label_text'] in res: results["snli"] += 1

# Jigsaw
print("Testing Safety (Jigsaw)...")
for item in tqdm(golden_test_set["jigsaw"]):
    user_input = f"Rule: {item['rule']}\nComment: {item['body']}\nDoes this comment violate the rule? Answer Yes or No."
    res = get_response("You are a forum moderator. Analyze if the comment violates the specific rule. Answer with only 'yes' or 'no'.", user_input)
    true_label = "yes" if item['rule_violation'] == 1 else "no"
    if true_label in res: results["jigsaw"] += 1

# Squad_v2
print("Testing Reading (SQuAD)...")
for item in tqdm(golden_test_set["squad"]):
    res = get_response("Answer the question based on context. Shortest answer possible. If no answer, say 'unanswerable'.", f"C: {item['context']}\nQ: {item['question']}", max_tokens=30)
    answers = [str(a).lower() for a in item['answers']]
    if not answers:
        if "unanswerable" in res: results["squad"] += 1
    else:
        if any(a in res for a in answers): results["squad"] += 1

# ==========================================
# Final report of Merged model on 3 scenarios
print("\n" + "★"*20 + " FINAL REPORT " + "★"*20)
print(f"Merged Model - SNLI Score:   {results['snli']/SAMPLE_SIZE:.2%}")
print(f"Merged Model - Jigsaw Score: {results['jigsaw']/SAMPLE_SIZE:.2%}")
print(f"Merged Model - SQuAD Score:  {results['squad']/SAMPLE_SIZE:.2%}")
print("★"*54)
