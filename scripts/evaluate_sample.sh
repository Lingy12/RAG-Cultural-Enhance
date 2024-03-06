python src/evaluator.py /home/shared_LLMs/gemma-2b-it prompt/prompt_eval.txt ./vector_store/wiki_filter_country_bge-large-en-v1.5 0.5 6 "sg_eval,us_eval,ph_eval" embed BAAI/bge-large-en-v1.5 

python src/evaluator.py /home/shared_LLMs/gemma-2b-it prompt/prompt_eval.txt ./vector_store/wiki_sg_exclusive_1_1_None_1_1.0_True.pkl/ 0.2 8 "sg_eval," tfidf


