from rag import RAG
from vector_store import TfidfStore
import time
from langchain_core.documents import Document
from evaluate import eval_rag
import sys
import os
from pathlib import Path
import transformers
transformers.set_seed(43)

def to_doc(text):
    return Document(page_content=text)

model_path = '/home/shared_LLMs/gemma-2b-it/'
prompt = './prompt/prompt_test.txt'
vector_store = os.path.normpath(sys.argv[1])
retrival_threshold = float(sys.argv[2])

# Build the TF-IDF vector store
start_time = time.time()
print('Loading vector store...')
vs = TfidfStore(top_k=5,saved_vs=vector_store)
print('Vector store loaded.')
end_time = time.time()


rag_model = RAG(model_path=model_path, vector_store=vs, prompt_template=prompt, device='cuda', verbose=1, retrieval_threshold=retrival_threshold)

datasets = ['sg_eval', 'us_eval', 'ph_eval']
# eval_lang = ['English']
num_prompt = 1

for dataset in datasets:
    for i in range(1, num_prompt + 1):
        eval_rag(rag=rag_model, dataset_name=dataset, prompt_index=i, eval_lang=['English'], eval_mode='zero_shot', model_name=str(Path(os.path.basename(vector_store)).with_suffix('')) + '_' + str(retrival_threshold))
