from rag import RAG
from vector_store import TfidfStore
import time
from langchain_core.documents import Document

def to_doc(text):
    return Document(page_content=text)

model_path = './models/gemma-2b-it/'
prompt = './prompt/prompt_test.txt'

random_texts = ['This first document is about how to study well in text mining.', 
                'The second document is about how to study in machine learning.', 
                'This is the third one', 
                "Is this the first document? It's about text mining."]
random_docs = list(map(lambda x: to_doc(x), random_texts))
num_docs = len(random_docs)
# Build the TF-IDF vector store
vs = TfidfStore(top_k=2)
start_time = time.time()
vs.build_vectorstore(random_docs)
end_time = time.time()

print(f"Building time for {num_docs} documents: {end_time - start_time} seconds")

rag_model = RAG(model_path=model_path, vector_store=vs, prompt_template=prompt, device='cuda')

results = rag_model.generate('What is the first document?')

print(results)

