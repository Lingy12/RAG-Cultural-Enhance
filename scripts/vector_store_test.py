# from vector_store import TfidfStore
# from langchain_core.documents import Document
#
# def to_doc(text):
#     return Document(page_content=text)
#
# vs = TfidfStore(top_k=2)
# text = ['This is the first document', 'This is the second document', 'This is the third one', 'Is this the first document?']
# docs = list(map(lambda x: to_doc(x), text))
# vs.build_vectorstore(docs)
#
# res = vs.query('This is the third one')
#
# print(res)

import random
import string
import time
from vector_store import TfidfStore
from langchain_core.documents import Document

def random_string(length=100):
    return ''.join(random.choice(string.ascii_lowercase + ' ') for _ in range(length))

def to_doc(text):
    return Document(page_content=text)

# Generate 100,000 random documents
# num_docs = 10
# random_texts = [random_string() for _ in range(num_docs)]
# random_docs = list(map(to_doc, random_texts))
random_texts = ['This is the first document', 'This is the second document', 'This is the third one', 'Is this the first document?']
random_docs = list(map(lambda x: to_doc(x), random_texts))
num_docs = len(random_docs)
# Build the TF-IDF vector store
vs = TfidfStore(top_k=2)
start_time = time.time()
vs.build_vectorstore(random_docs)
end_time = time.time()

print(f"Building time for {num_docs} documents: {end_time - start_time} seconds")

# Query the vector store
res = vs.query('This is the third one')
print(res)

