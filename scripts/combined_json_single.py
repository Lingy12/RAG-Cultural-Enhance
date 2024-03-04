from glob import glob
import os
from langchain_core.documents import Document
from logger_config import get_logger
import json
from tqdm import tqdm
import fire
from langchain.document_loaders.json_loader import JSONLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from typing import Iterable, List

logger = get_logger(__name__)

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in tqdm(array):
            jsonl_file.write(doc.json() + '\n')

def contains_any(s, words):
    return any(word in s for word in words)

def to_doc(text):
    return Document(page_content=text)

def combine_json(data_dir, name, filter_words: List[str]):
    
    if not os.path.exists('./processed_data'):
        os.makedirs('./processed_data', exist_ok=True)

    doc_lst = []

    raw_lst = glob(f'{data_dir}/**/*')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    logger.info('Total document list = {}'.format(len(raw_lst)))
    count = 0 
    for doc in tqdm(raw_lst):
        with open(doc, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            data = json.loads(line)
            
            if not contains_any(data['text'].lower(), filter_words):
                continue
            

            if len(data['text'].strip()) >= 200:
                splited_text = text_splitter.split_text(data['text'])
                doc_lst.extend(list(map(lambda x: to_doc(x), splited_text)))
                # doc_lst.append(to_doc(data['text']))
            count += 1
    logger.info('Total document without filter = {}'.format(count))
    logger.info('Total filtered document with split = {}'.format(len(doc_lst)))
    
    logger.info('Saving the document..')
    save_docs_to_jsonl(file_path=f'processed_data/{name}.jsonl', array=doc_lst)
    logger.info('Documents saved.')

if __name__ == "__main__":
    fire.Fire(combine_json)
