from rag import RAG
from vector_store import TfidfStore
import time
from langchain_core.documents import Document
from evaluate import eval_rag
import sys
import os
from pathlib import Path
import transformers
from transformers import set_seed
# transformers.set_seed(43)
import torch
def make_deterministic(seed):
    set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def to_doc(text):
    return Document(page_content=text)

make_deterministic(0)
model_path = '/home/shared_LLMs/gemma-2b-it/'
prompt = './prompt/prompt_eval.txt'
vector_store = os.path.normpath(sys.argv[1])
retrival_threshold = float(sys.argv[2])
top_k = int(sys.argv[3])

# Build the TF-IDF vector store
# start_time = time.time()
# print('Loading vector store...')
# vs = TfidfStore(top_k=5,saved_vs=vector_store)
# print('Vector store loaded.')
# end_time = time.time()


datasets = {'sg_eval': f'./vector_store/{vector_store}'}
            # 'us_eval': './vector_store/wiki_us_ex_filtered_1_1_None_1_1.0_True.pkl', 
            # 'ph_eval': './vector_store/wiki_ph_exclusive_1_1_None_1_1.0_True.pkl'}
# eval_lang = ['English']
num_prompt = 1

for dataset in datasets:
    for i in range(1, num_prompt + 1):
        print('Loading vector store...')
        vs = TfidfStore(top_k=top_k,saved_vs=datasets[dataset])
        print('Vector store loaded.')

        rag_model = RAG(model_path=model_path, vector_store=vs, prompt_template=prompt, retrieval_threshold=retrival_threshold, verbose=1)
        eval_rag(rag=rag_model, dataset_name=dataset, prompt_index=i, eval_lang=['English'], eval_mode='zero_shot', model_name=str(Path(os.path.basename(datasets[dataset])).with_suffix('')) + '_' + str(retrival_threshold) + '_' + str(top_k))
