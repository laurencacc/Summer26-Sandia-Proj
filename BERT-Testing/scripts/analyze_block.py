from transformers import AutoTokenizer, AutoModelForMaskedLM 
import torch
model_name = "/home/lecacca/VSCode Environment/BERT-Testing/bert_uncased_L-2_H-128_A-2"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

text = "Paris is the capital of [MASK]."

inputs = tokenizer(text, return_tensors="pt")

block0 = model.bert.encoder.layer[0]

activations = {}

# Hook function, will save module's outputs into activations
def save_activation(name):
    def hook(module, input, output):
        # remove raw tensor values typically used for training
        activations[name] = output.detach()
    return hook

print("start")
print(model.bert.encoder.layer[0].attention.self.query)
print(model.bert.encoder.layer[0].attention.self.key)
print(model.bert.encoder.layer[0].attention.self.value)

print(model.bert.encoder.layer[0].attention.output.dense)
print(model.bert.encoder.layer[0].intermediate.dense)
print(model.bert.encoder.layer[0].output.dense)
print("end")

# temporary
handles = []

# attach a hook to the query layer, when the layer runs then save its output under the name "query"
handles.append(block0.attention.self.query.register_forward_hook(save_activation("query")))
handles.append(block0.attention.self.key.register_forward_hook(save_activation("key")))
handles.append(block0.attention.self.value.register_forward_hook(save_activation("value")))

handles.append(block0.attention.output.dense.register_forward_hook(save_activation("attention_output_dense")))
handles.append(block0.intermediate.dense.register_forward_hook(save_activation("intermediate_dense")))
handles.append(block0.output.dense.register_forward_hook(save_activation("output_dense")))

with torch.no_grad():
    outputs = model(**inputs)

for name, tensor in activations.items():
    print(f"{name}: shape = {tensor.shape}")

for handle in handles:
    handle.remove()

