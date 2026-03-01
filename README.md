# Reddit Community Rules Classification with Context Enhanced QLoRA Fine Tuning
We utilize the context-augmentation techniques to further enhance the performance of the fine-tuned Qwen 2.5 model (based on QLoRA) in predicting Reddit comment compliance. Compared to pure QLoRA fine-tuning technique, we aim to address rule ambiguity in small training datasets and add dynamic data augmentation to further enhance the generalization capabilities of fine-tuned models.


# Current Updates of Projects (2026.03.01)
We selected three task scenarios: Jigsaw - Agile Community Rules Classification (https://www.kaggle.com/competitions/jigsaw-agile-community-rules), Squad_v2 (https://huggingface.co/datasets/rajpurkar/squad_v2), SNLI (https://huggingface.co/datasets/stanfordnlp/snli). These three scenarios represent distinct task types:

## Subjective Rule Judgment / Complex Semantic Alignment. 
In Jigsaw - Agile Community Rules Classification, the model must not only comprehend “rules” but also discern hidden malice in comments (e.g., microaggressions, malicious sarcasm). This tests the model's value alignment and fine-grained instruction adherence. The specific data logic is: starting from Agile community rules, the model establishes strict safety boundaries through a binary “Yes/No” decision-making process.

##  Long-Context Processing / Extractive Question Answering
The Squad v2 scenario severely tests the model's focus. The model must precisely locate answers within a context spanning over 1000 tokens. Additionally, the Squad v2 version incorporates some “unanswerable” data, placing higher demands on the model's ability to resist hallucinations (requiring it to refuse to answer when evidence is insufficient, rather than generating an unfounded response).

## Formal Logic / Semantic Entailment
The SNLI scenario involves the purest form of logical computation. The model must determine whether the relationship between the premise (P) and the hypothesis (H) is entailment, neutrality, or contradiction. This concerns neither common sense nor safety, but solely pure semantic derivation. Therefore, the model must detect logical shifts within minute lexical variations.
