from vector_store import TfidfStore
from glob import glob
import os
from langchain_core.documents import Document
from logger_config import get_logger
import json
from tqdm import tqdm
import fire
from langchain.document_loaders.json_loader import JSONLoader
from langchain_text_splitters import CharacterTextSplitter
from typing import Iterable
from pathlib import Path
import numpy as np
import time

logger = get_logger(__name__)

def load_docs_from_jsonl(file_path):
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file.readlines()):
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array
def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in tqdm(array):
            jsonl_file.write(doc.json() + '\n')

def to_doc(text):
    return Document(page_content=text)

def build_tf_idf(json_file, no_of_sample, output):
    
 
    logger.info('Loading data...')
    start = time.time()
    data = load_docs_from_jsonl(json_file)
    logger.info('Data loaded in {}s'.format(time.time() - start))

    subset = np.random.choice(data, size=no_of_sample, replace=False)
    save_docs_to_jsonl(subset, output)
if __name__ == "__main__":
    fire.Fire(build_tf_idf)
