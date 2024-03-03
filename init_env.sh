export PYTHONPATH="$PWD/eval_src:$PWD/src" 

conda create -n rag python==3.10

pip install -r requirements.txt
