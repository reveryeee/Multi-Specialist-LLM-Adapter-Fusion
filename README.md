# Multi-Specialist-LLM-Adapter-Fusion
In this project, we will explore the precision enhancement possibility of specialized domain performance by training and merging different distinct expert adapters and demonstrate the exceptional potential of LoRA Merging in resolving multi-task conflicts and mitigating "Catastrophic Forgetting" caused by LLM fine-tuning.

# Current Updates of Projects (2026.03.01)
We selected three task scenarios: Jigsaw - Agile Community Rules Classification (https://www.kaggle.com/competitions/jigsaw-agile-community-rules), Squad_v2 (https://huggingface.co/datasets/rajpurkar/squad_v2), SNLI (https://huggingface.co/datasets/stanfordnlp/snli). These three scenarios represent distinct task types:

## Subjective Rule Judgment / Complex Semantic Alignment. 
In Jigsaw - Agile Community Rules Classification, the model must not only comprehend “rules” but also discern hidden malice in comments (e.g., microaggressions, malicious sarcasm). This tests the model's value alignment and fine-grained instruction adherence. The specific data logic is: starting from Agile community rules, the model establishes strict safety boundaries through a binary “Yes/No” decision-making process.

##  Long-Context Processing / Extractive Question Answering
The Squad v2 scenario severely tests the model's focus. The model must precisely locate answers within a context spanning over 1000 tokens. Additionally, the Squad v2 version incorporates some “unanswerable” data, placing higher demands on the model's ability to resist hallucinations (requiring it to refuse to answer when evidence is insufficient, rather than generating an unfounded response).

## Formal Logic / Semantic Entailment
The SNLI scenario involves the purest form of logical computation. The model must determine whether the relationship between the premise (P) and the hypothesis (H) is entailment, neutrality, or contradiction. This concerns neither common sense nor safety, but solely pure semantic derivation. Therefore, the model must detect logical shifts within minute lexical variations.

## Our current benchmark results (2026.03.01)
We tested with 5 models:
1. Baseline Model (original Qwen2.5-7B);  
2. Jigsaw Specialist Model (Qwen2.5-7B fine-tuned on Jigsaw data);  
3. SNLI Specialist Model (Qwen2.5-7B fine-tuned on SNLI data);  
4. Squad Specialist Model (Qwen2.5-7B fine-tuned on Squad v2 data);  
5. Merged Model (combination of Jigsaw Specialist, SNLI Specialist, and Squad Specialist).

Our testing results are shown below:
| Task Scenario | Baseline Model | Jigsaw Specialist Model | SNLI Specialist Model | SQuAD Specialist Model | Merged Model |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SNLI (Logic)** | 88.60% | 87.40% | 92.00% | 80.20% | **86.80%** |
| **Jigsaw (Safety)** | 54.40% | 77.20% | 52.60% | 60.80% | **77.00%** |
| **SQuAD (Reading)** | 78.80% | 73.60% | 73.40% | 83.00% | **80.20%** |

From the experimental results, we can observe that the baseline model exhibits significant shortcomings in the jigsaw scenario. Furthermore, while different expert models demonstrate improved performance in their specific target scenarios, their capabilities decline in other scenarios. 
However, the merged model achieves multi-domain expertise while successfully mitigating catastrophic forgetting, as it shows best overall performance compared to other 4 models.
