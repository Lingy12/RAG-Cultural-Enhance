# Retrival Augmented Generation (RAG) to enhance cultural reasoning and understanding

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

## Download and extract wikipedia data

```bash
bash scripts/download_and_extract_wiki.sh
python scripts/combine_json.py wiki_filtered True # create filtered data

# processed_data/wiki_filtered.jsonl will be created
```

## Apply post processing (Optional, only for running tf-idf algorithm)
```bash
python scripts/sample_subset.py processed_data/wiki_filter_country.jsonl 50000 processed_data/wiki_filter_random_50k.jsonl

## just an example, change '100000' to desired sample for running tf-idf
```

## Build Vector Store
```bash
python scripts/build_tf_idf_vs.py processed_data/wiki_filter_random_50k.jsonl 1 1 None 1 50000 True

# a file under vector_store/ will created, the above command will create ./vector_store/wiki_filter_random_50k_1_1_None_1_50000_True.pkl
# refer to the code for vectorizer config (scripts/build_tf_idf_vs.py)
```
