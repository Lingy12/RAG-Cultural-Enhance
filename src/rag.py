from transformers import AutoTokenizer, AutoModelForCausalLM
from vector_store import BaseVectorStore
import re
from logger_config import get_logger
import torch
import logging

logger = get_logger(__name__)
class RAG:

    def __init__(self, model_path: str, vector_store: BaseVectorStore, prompt_template:str, retrieval_threshold=0.4, device='cpu', verbose=0):
        if verbose == 1:
            logger.setLevel(logging.ERROR)
        logger.info('Initalizing RAG...')
        self.retrieval_threshold = retrieval_threshold
        self.vector_store = vector_store
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32 if device == 'cpu' else torch.float16, device_map=device)
        
        logger.info('RAG initailized.')

        with open(prompt_template, 'r') as f:
            self.prompt_template = f.read()

    def generate(self, query, max_new_tokens=128, show_ori=False):
        pattern = r"Question:\n(.*?)\n\nChoices:"
        match = re.search(pattern, query, re.I) # check seaeval prompt, query question only
        # logger.info(match.group(1))
        if match:
        # Step 1: retrival
            query_result = self.vector_store.query(match.group(1).strip())
            logger.info('Running seaeval, extract the question only..')
            logger.info('Question: ' + match.group(1).strip())
        else:
            query_result = self.vector_store.query(query)
        
        # can add logic to filter
        logger.info('Retrieval result:')
        logger.info(query_result)
        scores = list(map(lambda x: x[1], query_result))
        filtered_retrival = list(filter(lambda x: x[1] > self.retrieval_threshold, query_result))
        if len(filtered_retrival) > 0:
            retrival_string = '\n'.join(list(map(lambda x: x[0].page_content, filtered_retrival)))
        else:
            retrival_string = ''
        logger.info('Retrival string: ')
        logger.info(retrival_string)
        logger.info('Simiarity score: {}'.format(scores))
        
    
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
        if len(retrival_string) > 0:
            outputs[0] += '[RAG]'
        logger.info('model answer: ' + outputs[0])
        return outputs


