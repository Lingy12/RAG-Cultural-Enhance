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
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

logger = get_logger(__name__)

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in tqdm(jsonl_file.readlines()):
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

# Function to count unique n-grams
def count_unique_ngrams(texts, n):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    X = vectorizer.fit_transform(texts)
    return len(vectorizer.get_feature_names_out())


def to_doc(text):
    return Document(page_content=text)

def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove all punctuation
    text = text.lower()  # Convert to lowercase
    return text

def check_data(path):
    ds = load_docs_from_jsonl(path)

    print('Number of documents: {}'.format(len(ds)))

    text_data = list(map(lambda x: preprocess_text(x.page_content), ds))
    max_length = max(list(map(lambda x: len(x), text_data)))

    print(f"Maximum length of the documents: {max_length}")

    unique_1gram = count_unique_ngrams(text_data, 1)
    unique_2gram = count_unique_ngrams(text_data, 2)
    unique_3gram = count_unique_ngrams(text_data, 3)

    print(f"Unique 1-gram: {unique_1gram}")
    print(f"Unique 2-gram: {unique_2gram}")
    print(f"Unique 3-gram: {unique_3gram}")

if __name__ == "__main__":
    fire.Fire(check_data)




    

    
