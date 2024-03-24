import os
from pathlib import Path
from vector_store import TfidfStore
import time
import fire
from logger_config import get_logger
import json
from scripts.build_tf_idf_vs import load_docs_from_jsonl, to_doc

logger = get_logger(__name__)
def build_tf_idf(json_file, params_file, max_features, use_idf: bool):
    if not os.path.exists('./vector_store'):
        os.makedirs('./vector_store', exist_ok=True)

    logger.info('Loading data...')
    start = time.time()
    data = load_docs_from_jsonl(json_file)
    logger.info('Data loaded in {}s'.format(time.time() - start))

    # Load parameters from the params_file
    all_params = []
    with open(params_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            all_params.append(json.loads(line))
    
    logger.info('Total parameters = {}'.format(len(all_params)))
    for params in all_params:
        n_min = params.get('min_n', 1)
        n_max = params.get('max_n', 1)
        max_df = params.get('max_df', 1.0)
        min_df = params.get('min_df', 1)

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
        logger.info('Saved vector store: ' + output_name)

if __name__ == '__main__':
    fire.Fire(build_tf_idf)
