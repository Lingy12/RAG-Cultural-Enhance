from rag import RAG
from vector_store import TfidfStore
import time
from langchain_core.documents import Document
from evaluate import eval_rag
import transformers

transformers.set_seed(43)
def to_doc(text):
    return Document(page_content=text)

model_path = '/home/shared_LLMs/gemma-2b-it'
prompt = './prompt/prompt_test.txt'
vector_store = './vector_store/wiki_filter_test_1_1_None_1_50000_True.pkl'

# Build the TF-IDF vector store
start_time = time.time()
print('Loading vector store...')
vs = TfidfStore(top_k=5,saved_vs=vector_store)
print('Vector store loaded.')
end_time = time.time()
# vs.build_vectorstore(random_docs)
# end_time = time.time()

print(f"Loading time is : {end_time - start_time} seconds")

rag_model = RAG(model_path=model_path, vector_store=vs, prompt_template=prompt, device='cuda', verbose=0, retrieval_threshold=0.1)

while True:
    query = input('Enter the query:')
    answer = rag_model.generate(query, show_ori=False, max_new_tokens=64)
    print('Model Response: ' + answer[0])