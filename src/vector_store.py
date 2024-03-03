import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from logger_config import get_logger
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import List

logger = get_logger(__name__)

class MyTFIDFRetriever(TFIDFRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[tuple]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc
        return_docs = [(self.docs[i], results[i]) for i in results.argsort()[-self.k :][::-1]]

        return return_docs

class BaseVectorStore:
    def build_vectorstore(self, docs):
        raise Exception('Show call implemented version.')
    
    def export(self, path):
        raise Exception('Show call implemented version.')
    
    def query(self, query):
        raise Exception('Show call implemented version.')

class TfidfStore(BaseVectorStore):
    
    def __init__(self, top_k, vectorizer_kwargs={}, saved_vs=None):
        self.top_k = top_k
        self.vectorizer_kwargs = vectorizer_kwargs

        if saved_vs:
            logger.info('Loading saved vector store')
            self.vectorstore = MyTFIDFRetriever.load_local(saved_vs)
            self.vectorstore.k = self.top_k
            logger.info('Vector store loaded')
            self.is_load_from_local = True
        else:
            self.is_load_from_local = False

    def build_vectorstore(self, docs):
        # tfidf_matrix = self.vectorizer.fit_transform(docs)
        if self.is_load_from_local:
            raise Exception('You have loaded from local vector store. Please query the vector store')
        logger.info('Building vector stores with {} documents'.format(len(docs)))
        self.vectorstore = MyTFIDFRetriever.from_documents(docs, tfidf_params=self.vectorizer_kwargs)
        self.vectorstore.k = self.top_k

    def export(self, path):
        self.vectorstore.save_local(folder_path=path)
    
    def query(self, query_str):
        results = self.vectorstore.get_relevant_documents(query_str)
        return results[:self.top_k]

    def get_vocabulary(self):
        return self.vectorstore.vectorizer.vocabulary_


class EmbeddingStore(BaseVectorStore):
    
    def __init__(self, top_k, model_name, saved_vs, clean=False):
        logger = get_logger('embedding-vectorstore')
        self.top_k = top_k
        self.embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
        self.saved_vs = saved_vs
        self.allow_build = True
        if os.path.exists(saved_vs) and not clean:
            logger.warning('Vector store database already exists. Please delete it or use it without rebuild.')
            self.allow_build = False
        if os.path.exists(saved_vs) and clean:
           shutil.rmtree(saved_vs) 
        logger.info('Loading saved vector store')
        self.vectorstore = Chroma(embedding_function=self.embedding_function, persist_directory=saved_vs)
        self.vectorstore.k = self.top_k
        logger.info('Vector store loaded')

    def build_vectorstore(self, docs):
        # tfidf_matrix = self.vectorizer.fit_transform(docs)
        if self.allow_build:
            logger.info('Building vector stores with {} documents'.format(len(docs)))
            self.vectorstore = Chroma.from_documents(docs, self.embedding_function, persist_directory=self.saved_vs)
            logger.info('Vector store built at {}'.format(self.saved_vs))
        else:
            raise Exception('Should not build a new vector store.')

    def export(self, path):
        logger.warning('Vector store is saved at {} automatically'.format(self.saved_vs))
    
    def query(self, query_str):
        results = self.vectorstore.similarity_search_with_relevance_scores(query_str)[:self.top_k]
        return results


        



        



    
