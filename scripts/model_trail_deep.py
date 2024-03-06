from rag import RAG
from vector_store import EmbeddingStore
import time
from langchain_core.documents import Document
from evaluate import eval_rag

def to_doc(text):
    return Document(page_content=text)

model_path = '/home/shared_LLMs/gemma-2b-it/'
prompt = './prompt/prompt_test.txt'
vector_store = './vector_store/wiki_filter_country_bge-large-en-v1.5'

# Build the TF-IDF vector store
start_time = time.time()
print('Loading vector store...')
vs = EmbeddingStore(top_k=2,saved_vs=vector_store, model_name='BAAI/bge-large-en-v1.5')
print('Vector store loaded.')
end_time = time.time()
# vs.build_vectorstore(random_docs)
# end_time = time.time()

print(f"Loading time is : {end_time - start_time} seconds")

rag_model = RAG(model_path=model_path, vector_store=vs, prompt_template=prompt, device='cuda', verbose=0)

while True:
    query = input('Enter the query:')
    answer = rag_model.generate(query)
    print('Model Response: ' + answer[0])
