# Retrival Augmented Generation (RAG) to enhance cultural understanding

## Set up environment
```bash
export PYTHONPATH=$PWD/src:$PWD/eval_src 

cat "export PYTHONPATH=$PWD/src:$PWD/eval_src" >> ~/.bashrc 
```

```bash
conda create -n rag python==3.10

pip install -r requirements.txt

conda activate rag
```

## Reproducibility

We know that the sampling could help to boost the generation result from LLM, however, to have a fair comparasion and consistent result, we disable the sampling for model generation and set determinitic for CUDA and transformers package. 

```json
## Generation config
{
  "_from_model_config": true,
  "bos_token_id": 2,
  "eos_token_id": 1,
  "pad_token_id": 0,
  "transformers_version": "4.38.0.dev0"
}
```

```python
from transformers import set_seed
import torch

def make_deterministic(seed):
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

make_deterministic(0)
```

## Download and extract wikipedia data

```bash
bash scripts/download_and_extract_wiki.sh
python scripts/combine_json.py wiki_filtered True # create filtered data

# processed_data/wiki_filtered.jsonl will be created
```

## Build Vector Store
```bash
python scripts/build_tf_idf_vs.py processed_data/wiki_filter_random_50k.jsonl 1 1 None 1 50000 True

# a file under vector_store/ will created, the above command will create ./vector_store/wiki_filter_random_50k_1_1_None_1_50000_True.pkl
# refer to the code for vectorizer config (scripts/build_tf_idf_vs.py)
```

## Running Evaluation
```bash
# Evalute tfidf
python src/evaluator.py ./models/gemma-2b-it prompt/prompt_eval.txt ./vector_store/wiki_sg_exclusive_1_1_None_1_1.0_True.pkl/ 0.2 8 "sg_eval," tfidf

# Evaluate embedding

python src/evaluator.py ./models/gemma-2b-it prompt/prompt_eval.txt \
                      ./vector_store/wiki_filter_country_bge-large-en-v1.5 0.5 6 \
                        "sg_eval," embed BAAI/bge-large-en-v1.5 # no rerank

python src/evaluator.py ./models/gemma-2b-it prompt/prompt_eval.txt \
                      ./vector_store/wiki_filter_country_bge-large-en-v1.5 0.4 64 \
                          "sg_eval," embed BAAI/bge-large-en-v1.5 --verbose 0 \
                            --need_rerank --rerank_sample 8 # with reranking
```
