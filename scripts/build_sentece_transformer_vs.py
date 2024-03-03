from vector_store import EmbeddingStore
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
import time

logger = get_logger(__name__)

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file.readlines()):
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

def to_doc(text):
    return Document(page_content=text)

def build_vector_store(json_file, model_name):
    
    if not os.path.exists('./vector_store'):
        os.makedirs('./vector_store', exist_ok=True)
    
    logger.info('Loading data...')
    start = time.time()
    data = load_docs_from_jsonl(json_file)
    logger.info('Data loaded in {}s'.format(time.time() - start))
    
    logger.info(model_name)
    save_name = str(Path(os.path.basename(json_file)).with_suffix('')) + '_' + model_name
    embedding_store = EmbeddingStore(top_k=20, model_name=model_name, saved_vs=os.path.join('./vector_store', save_name))
    logger.info('Vector store initialized.')
    
    start = time.time()
    logger.info('Building embedding vector store.')
    embedding_store.build_vectorstore(data)
    logger.info('Built vector store.')
    runtime = time.time() - start
    logger.info('Built time: {}s'.format(runtime))

if __name__ == "__main__":
    fire.Fire(build_vector_store)
