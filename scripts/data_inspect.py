from glob import glob
import os
from langchain_core.documents import Document
from logger_config import get_logger
import json
from tqdm import tqdm
import fire
from langchain.document_loaders.json_loader import JSONLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from typing import Iterable

logger = get_logger(__name__)


def contains_any(s, words):
    return any(word in s for word in words)

filter_words = ['singapore', 'united state', 'u.s.', 'philippines']

def check_data(data_dir):
    

    raw_lst = glob(f'{data_dir}/**/*')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    logger.info('Total document list = {}'.format(len(raw_lst)))
    count = 0 
    count_split = 0
    all_count = 0
    for doc in tqdm(raw_lst):
        with open(doc, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            data = json.loads(line)
            all_count += 1
            if not contains_any(data['text'].lower(), filter_words):
                continue

            if len(data['text'].strip()) >= 200:
                splited_text = text_splitter.split_text(data['text'])
                # doc_lst.append(to_doc(data['text']))
                count_split += len(splited_text)
            count += 1
    logger.info('Total document without filter = {}'.format(all_count))
    logger.info('Total document with target words = {}'.format(count))
    logger.info('Total filtered document with split = {}'.format(count_split))
    

if __name__ == "__main__":
    fire.Fire(check_data)
