from transformers import AutoTokenizer, AutoModelForCausalLM
from vector_store import BaseVectorStore
import re
from logger_config import get_logger
import torch
import logging
from langchain.retrievers.bm25 import BM25Retriever

logger = get_logger(__name__)

def reranking(retrived_doc, rerank_sample, query):
    if len(retrived_doc) <= rerank_sample:
        logger.warning('Retrieved document already less than the rerank_sample')
        return list(map(lambda x: (x, -1), retrived_doc))
    retriver = BM25Retriever.from_documents(retrived_doc)
    result = retriver.get_relevant_documents(query)
    return list(map(lambda x: (x, -1), result[:rerank_sample])) # assign a place holder
    
class RAG:

    def __init__(self, model_path: str, vector_store: BaseVectorStore, prompt_template:str, retrieval_threshold=0.4, device='cpu', verbose=0, need_rerank=False, rerank_sample=8):
        if verbose == 0: # silent
            logger.setLevel(logging.ERROR)
        logger.info('Initalizing RAG...')
        self.retrieval_threshold = retrieval_threshold
        self.vector_store = vector_store
        self.device = device
        
        self.need_rerank = need_rerank
        self.rerank_sample = rerank_sample
        if self.need_rerank:
            logger.warning('Using reranking for RAG processing')
            logger.warning('Sample after rerank = {}'.format(rerank_sample))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32 if device == 'cpu' else torch.float16, device_map=device)
        
        logger.info('RAG initailized.')

        with open(prompt_template, 'r') as f:
            self.prompt_template = f.read()

    def generate(self, query, max_new_tokens=128, show_ori=False):
        # pattern = r"Question:\n(.*?)\n\nChoices:"
        # match = re.search(pattern, query, re.I) # check seaeval prompt, query question only
        # logger.info(match.group(1))
        # if match:
        # Step 1: retrival
            # query_result = self.vector_store.query(match.group(1).strip())
            # logger.info('Running seaeval, extract the question only..')
            # logger.info('Question: ' + match.group(1).strip())
        # else:
        query_result = self.vector_store.query(query)
        
        if len(query) < self.rerank_sample and self.need_rerank:
            raise Exception('Please specify rerank_sample < top_k of vector store')
        
        # can add logic to filter
        logger.info('Retrieval result:')
        logger.info(query_result)
        scores = list(map(lambda x: x[1], query_result))
        logger.info('Retriveval scores = {} (before reranking)'.format(scores))
        filtered_retrival = list(filter(lambda x: x[1] > self.retrieval_threshold, query_result))
    
        if self.need_rerank:
            filtered_retrival = reranking(list(map(lambda x: x[0], filtered_retrival)), self.rerank_sample, query)
            logger.info('Reranking finished.')
        # print(filtered_retrival)
        if len(filtered_retrival) > 0:
            retrival_string = '\n'.join(list(map(lambda x: x[0].page_content, filtered_retrival)))
        else:
            retrival_string = ''
            
        logger.info('Retrival string: ')
        logger.info(retrival_string)
            
           
        llm_input = self.prompt_template.format(retrival_string, query)
        llm_no_rag_input = self.prompt_template.format('', query)
        # self.logger.info('LLM inputs: ')
        # self.logger.info(llm_input)
        
        if show_ori:
            input_ids = self.tokenizer(llm_no_rag_input, return_tensors='pt').to(self.device)
            outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
        
            outputs = outputs[:,input_ids.input_ids.shape[-1]:] # mask input
            outputs = self.tokenizer.batch_decode(outputs)
            logger.info('[No RAG]' + outputs[0])

        # llm inference
        input_ids = self.tokenizer(llm_input, return_tensors='pt').to(self.device)
        outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
        
        outputs = outputs[:,input_ids.input_ids.shape[-1]:] # mask input
        outputs = self.tokenizer.batch_decode(outputs)
        
        if self.need_rerank and len(filtered_retrival) < self.rerank_sample and len(filtered_retrival) != 0:
            outputs[0] += '[RERANK]' # add tag if the reranking triggered.
        if len(retrival_string) > 0:
            outputs[0] += '[RAG]'
     
        logger.info('model answer: ' + outputs[0])
        return outputs


