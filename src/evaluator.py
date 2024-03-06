import torch
import fire
from rag import RAG
from vector_store import TfidfStore, EmbeddingStore
from langchain_core.documents import Document
from evaluate import eval_rag
import os
from pathlib import Path
from transformers import set_seed
from logger_config import get_logger

logger = get_logger(__name__)

def make_deterministic(seed):
    set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def to_doc(text):
    return Document(page_content=text)

make_deterministic(0)

def evaluate(model_path, prompt, vector_store, retrieval_threshold, top_k, eval_data, eval_mode, embed_name=None, verbose=0, device='cuda', need_rerank=False, rerank_sample=8):
    # model_path = '/home/shared_LLMs/gemma-2b-it/'
    # prompt = './prompt/prompt_eval.txt'
    # vector_store = os.path.normpath(sys.argv[1])
    # retrival_threshold = float(sys.argv[2])
    # top_k = int(sys.argv[3])
    # model_name = sys.argv[4]
    model_path_n = os.path.basename(os.path.normpath(model_path))
    prompt_n = Path(os.path.basename(prompt)).with_suffix('')
    vector_store_n = os.path.basename(os.path.normpath(vector_store))
    # Build the TF-IDF vector store
    # start_time = time.time()
    # print('Loading vector store...')
    # vs = TfidfStore(top_k=5,saved_vs=vector_store)
    # print('Vector store loaded.')
    # end_time = time.time()
    run_tag = '-'.join([str(model_path_n), str(prompt_n), str(vector_store_n), str(retrieval_threshold), str(top_k), str(need_rerank), str(rerank_sample)]) 
    logger.info(run_tag)
    # datasets = ['sg_eval', 'us_eval', 'ph_eval']
                # 'us_eval': './vector_store/wiki_us_ex_filtered_1_1_None_1_1.0_True.pkl', 
                # 'ph_eval': './vector_store/wiki_ph_exclusive_1_1_None_1_1.0_True.pkl'}
    # eval_lang = ['English']
    num_prompt = 1

    for dataset in eval_data:
        for i in range(1, num_prompt + 1):
            logger.info('Loading vector store...')
            
            if eval_mode == 'embed':
                if embed_name is None:
                    raise Exception('You must specify a embed name for your vector store for embed mode.')
                vs = EmbeddingStore(top_k=top_k, saved_vs=vector_store, model_name=embed_name) 
            elif eval_mode == 'tfidf':
                vs = TfidfStore(top_k=top_k, saved_vs=vector_store)
            else:
                raise Exception("Not a valid Eval mode. Choose ['embed', 'tfidf']")
            logger.info('Vector store loaded.')
            rag_model = RAG(model_path=model_path, vector_store=vs, prompt_template=prompt, retrieval_threshold=retrieval_threshold, verbose=verbose, device=device, need_rerank=need_rerank, rerank_sample=rerank_sample)
            logger.info('Model initalized.')
            eval_rag(rag=rag_model, dataset_name=dataset, prompt_index=i, eval_lang=['English'], eval_mode='zero_shot', 
                     model_name=run_tag)


if __name__ == "__main__":
    fire.Fire(evaluate)
