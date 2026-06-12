# Import Hugging Face classes
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
## Torch
## Tokenizer/model loader from transformers
## String model selector for google/bert_uncased_L-2_H-128_A-2
model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
# Load the tokenizer
## Hugging Face .from_pretrained(...) mechanism

tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenzier turns text into tokens, token IDs, tensors
model = AutoModelForMaskedLM.from_pretrained(model_name)
# model takes tensors and outputs predictions 

# Load the pre-trained model
# Put the model in eval mode
model.eval()

## Text must include [MASK]
text = "Paris is the capital of [MASK]."

# Tokenize a sequence with [MASK]
## Convert text to model inputs

inputs = tokenizer(text, return_tensors="pt")
# converts sentence into PyTorch tensors

# Run inference
## Run the model with gradients off
with torch.no_grad():
    outputs = model(**inputs)
# No gradients since no training needed

# Print the top predicted tokens at the mask
## Locate mask position (token index)
## Extract logits at that mask position
logits = outputs.logits
# typically (batch size, sequence length, vocab size)

## Get top-k vocabulary items (MLM head?)
mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
mask_index = mask_positions[0,1]
# Finds where [MASk] is in sequence

mask_logits = logits[0, mask_index, :]
# makes a vector, one score per vocabulary item

top_k=5
top_values, top_indices = torch.topk(mask_logits, top_k)
# Find 5-highest-scoring vocab entries

predicted_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())
# model predicts IDs, this turns numbers back into token strings

print(f"Input: {text}")
print("Top Predictions:")
for rank, (token,score) in enumerate(zip(predicted_tokens, top_values.tolist()), start=1):
    print(f"{rank}. {token} (logit={score:.4f})")