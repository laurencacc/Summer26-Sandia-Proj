from transformers import AutoModelForMaskedLM
# Load model and print it
model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()
# Inspect attributes of loaded model
# Locate - embeddings, encoder, list of layer/blocks
## model -> bert -> encoder -> layer[0]
## model -> bert -> encoder -> layer[1]
print("model:")
print(model)

print("model.bert:")
print(model.bert)

print("model.bert.encoder:")
print(model.bert.encoder)

print("model.bert.encoder.layer[0]:")
print(model.bert.encoder.layer[0])

print("model.bert.encoder.layer[1]:")
print(model.bert.encoder.layer[1])
