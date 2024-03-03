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

def build_tf_idf(json_file, n_min, n_max, max_features, min_df, max_df, use_idf:bool):
    
    if not os.path.exists('./vector_store'):
        os.makedirs('./vector_store', exist_ok=True)
    
    logger.info('Loading data...')
    start = time.time()
    data = load_docs_from_jsonl(json_file)
    logger.info('Data loaded in {}s'.format(time.time() - start))
    
    vectorizer_config = {"ngram_range": (n_min, n_max),
                         "max_df": max_df,
                         "min_df": min_df,
                         "stop_words": "english",
                         "max_features": max_features,
                         "use_idf": use_idf}
    logger.info("Vectorizer config: ")
    logger.info(vectorizer_config)
    tf_idf_store = TfidfStore(top_k=20, vectorizer_kwargs=vectorizer_config)
    logger.info('Vector store initialized.')
    
    start = time.time()
    logger.info('Building Tf-IDF vector store.')
    tf_idf_store.build_vectorstore(data)
    logger.info('Built vector store.')
    runtime = time.time() - start
    logger.info('Built time: {}s'.format(runtime))
    logger.info('Number of vocabulary: ' + str(len(tf_idf_store.get_vocabulary())))
    # logger.info(tf_idf_store.get_vocabulary())
    output_name = str(Path(os.path.basename(json_file)).with_suffix('')) + '_' + '_'.join([str(n_min), str(n_max), str(max_features), str(min_df), str(max_df), str(use_idf)]) + '.pkl'
    tf_idf_store.export(os.path.join('./vector_store', output_name))
    logger.info('Saved vector store')

if __name__ == "__main__":
    fire.Fire(build_tf_idf)
