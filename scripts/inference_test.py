# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("models/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("models/gemma-2b-it", device_map="cuda", torch_dtype=torch.float16)

input_text = "What is Retrival Argumented Generation?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_length=100)
print(tokenizer.decode(outputs[0]))



