from scripts.compare_output import compare_files
import fire
import os
from pathlib import Path
import glob
import json
import pandas as pd

baseline_file = 'log/baseline/sg_eval/gemma-2b-it_p1.json'

def consolidate_exp_folder(folder, output_path=None):
    prediction_files =  [f for f in glob.glob(os.path.join(folder, '*.json')) if 'score' not in f]
    output_table = {"run_name": [], "revise_count": [], 
                    "rag_error": [], "not_trigger_rag": [], 
                    "total_accuracy": [], "vector_store_name":[], "retrieval_threshold":[], 
                    "retrieval_number":[], "need_rerank":[], "doc_final": [], "model_type": [], 
                    "meta_data": [], 
                    "vector_store type": []}
    embedding_table = {"run_name": [], "revise_count": [], 
                    "rag_error": [], "not_trigger_rag": [], 
                    "total_accuracy": [], "vector_store_name":[], "retrieval_threshold":[], 
                    "retrieval_number":[], "need_rerank":[], "doc_final": [], "model_type": [], 
                    "meta_data": [], 
                    "vector_store type": []}
    tf_idf_table = {"run_name": [], "revise_count": [], 
                    "rag_error": [], "not_trigger_rag": [], 
                    "total_accuracy": [], "vector_store_name":[], "n_min":[], "n_max": [] ,
                     "max_features":[], "min_df": [], "max_df": [], "use_idf": [],
                    "retrieval_threshold":[], 
                    "retrieval_number":[], "need_rerank":[], "doc_final": [], "model_type": [], 
                    "vector_store type": []
        
    }
    for pred in prediction_files:
        compare_res = compare_files(baseline_file, pred)
        
        with open(str(Path(pred).with_suffix('')) + '_score.json') as f:
            accuracy = json.load(f)['accuracy']
            
        output_table['run_name'].append(os.path.basename(pred))
        output_table['revise_count'].append(compare_res['revise_count'])
        output_table['rag_error'].append(compare_res['rag_err'])
        output_table['not_trigger_rag'].append(compare_res['not_trigger_rag'])
        output_table['total_accuracy'].append(accuracy)
        
        run_name = str(Path(os.path.basename(pred)).with_suffix(''))
        
        segments = run_name.split('-')
        
        if len(segments) == 9:
            output_table['vector_store type'].append('tf-idf')
            vector_store_name = segments[4]  
            retrieval_threshold = segments[5]
            retrieval_number = segments[6]
            need_rerank = segments[7]
            doc_final = segments[8].split('_')[0]
            
            store_meta = vector_store_name.split('_')
            meta_data = json.dumps({"dataset": '_'.join(store_meta[:3]), 
                         "n_min": store_meta[3], "n_max": store_meta[4],
                         "max_features": store_meta[5], "min_df": store_meta[6], "max_df": store_meta[7], "use_idf": store_meta[8]})
        else:
            output_table['vector_store type'].append('embeddings')
            vector_store_name = '-'.join(segments[4:8])
            retrieval_threshold = segments[8]
            retrieval_number = segments[9]
            need_rerank = segments[10]
            doc_final = segments[11].split('_')[0]
            
            # print(store_meta)
            meta_data = json.dumps({})
        model_type = '-'.join(segments[:3])
        prompt = segments[3]
        
        for k in list(output_table.keys())[5:12]:
            output_table[k].append(eval(k))
            
        if output_table['vector_store type'][-1] == 'tf-idf':
            for k in tf_idf_table.keys():
                if k in output_table:
                    tf_idf_table[k].append(output_table[k][-1])
                else:
                    tf_idf_table[k].append(json.loads(meta_data)[k])
        else:
            for k in embedding_table.keys():
                embedding_table[k].append(output_table[k][-1])
    data = pd.DataFrame.from_dict(output_table)
    embedding = pd.DataFrame.from_dict(embedding_table)
    tf_idf = pd.DataFrame.from_dict(tf_idf_table)
    print(data)
    print(embedding)
    print(tf_idf)
    if output_path:
        writer = pd.ExcelWriter(output_path, engine='openpyxl')
        data.to_excel(writer, sheet_name='overall_results')
        embedding.to_excel(writer, sheet_name='embedding results')
        tf_idf.to_excel(writer, sheet_name='tf-idf results')
        writer.close()
if __name__ == '__main__':
    fire.Fire(consolidate_exp_folder)