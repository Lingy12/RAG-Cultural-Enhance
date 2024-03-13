import json
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import words



threshold_coefficient = 0.01

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

texts = [item['page_content'] for item in data]
threshold = int(threshold_coefficient * len(texts))

tokenized_texts = [[word for word in word_tokenize(text.lower()) if word in english_words and word not in stop_words and word not in string.punctuation and not word.startswith("\\u")] for text in texts]

ngram_ranges = [1, 2, 3]

results = []

for nrange in ngram_ranges:
    result = {"min_n": nrange, "max_n": nrange, "max_df_coefficient": threshold_coefficient}
    n_grams = []
    for text in tokenized_texts:
        n_grams.extend(list(ngrams(text, nrange)))
    df = Counter(n_grams)
    filtered_ngrams = [ngram for ngram, count in df.items() if count <= threshold]
    result["ngrams"] = filtered_ngrams
    results.append(result)

with open("../sg_data/sg_ngram_eval.jsonl", "w", encoding='utf-8') as write_file:
    for r in results:
        json_string = json.dumps(r)
        write_file.write(json_string + "\n")
