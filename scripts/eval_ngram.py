import json
import nltk
import string
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import words



threshold_coefficients = [0.001, 0.01, 0.1]

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')
english_words = set(words.words())
stop_words = set(stopwords.words('english'))

data = []
num_pages = 0
with open('../sg_data/wiki_sg_exclusive.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))
        num_pages += 1
file.close()

results = []
for threshold_coefficient in tqdm(threshold_coefficients):
    texts = [item['page_content'] for item in data]
    threshold = int(threshold_coefficient * len(texts))

    # tokenized_texts = [[word for word in word_tokenize(text.lower()) if word in english_words and word not in stop_words and word not in string.punctuation and not word.startswith("\\u")] for text in texts]

    ngram_ranges = [1, 2, 3]


    for nrange in tqdm(ngram_ranges):
        for max_n in tqdm(ngram_ranges):
            if max_n < nrange:
                continue
            result = {"min_n": nrange, "max_n": max_n, "max_df": threshold_coefficient}
            # n_grams = []
            # for text in tokenized_texts:
                # n_grams.extend(list(ngrams(text, nrange)))
            # df = Counter(n_grams)
            # filtered_ngrams = [ngram for ngram, count in df.items() if count <= threshold]
            # result["ngrams"] = filtered_ngrams
            results.append(result)

with open("../sg_data/sg_ngram_eval.jsonl", "w", encoding='utf-8') as write_file:
    for r in results:
        json_string = json.dumps(r)
        write_file.write(json_string + "\n")
