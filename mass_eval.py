from src import evaluator
from glob import glob
import os

model = '/home/shared_LLMs/gemma-2b'
vss = glob(os.path.join('./vector_store/', '*.pkl'))
print(vss)
retrieval_thresholds = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
top_ks = [1, 2, 4, 8]

print('Total params = {}'.format(len(vss) * len(retrieval_thresholds) * len(top_ks)))
for vs in vss:
    for retrieval_threshold in retrieval_thresholds:
        for top_k in top_ks:
            evaluator.evaluate(model_path=model, prompt='./prompt/prompt_eval.txt', 
                   vector_store=vs, retrieval_threshold=retrieval_threshold, top_k=top_k, 
                               eval_data=['sg_eval'], eval_mode='hidden_test')
