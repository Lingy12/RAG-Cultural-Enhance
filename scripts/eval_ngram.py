import json
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

data = []
num_pages = 0
with open('./sg_data/wiki_sg_exclusive.jsonl', 'r') as file:
    for line in file:
        data.append(json.loads(line))
        num_pages += 1

texts = [item['page_content'] for item in data]

tokenized_texts = [' '.join([word for word in word_tokenize(text.lower()) if word not in stop_words]) for text in texts]

ngram_ranges = [(1, 1), (2, 2), (3, 3)]

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

best_ngram_range = None
best_silhouette_score = -1
best_kmeans_model = None

for ngram_range in ngram_ranges:
    print("Trying n-gram range:", ngram_range)
    avg_silhouette_score = 0

    for train_indices, val_indices in k_fold.split(tokenized_texts):
        X_train, X_val = np.array(tokenized_texts)[train_indices], np.array(tokenized_texts)[val_indices]

        vectorizer = CountVectorizer(ngram_range=ngram_range)
        X_train = vectorizer.fit_transform(X_train)

        kmeans = KMeans(n_clusters=int(num_pages * 0.8 * 0.05), random_state=42)
        kmeans.fit(X_train)

        X_val_transformed = vectorizer.transform(X_val)
        labels = kmeans.predict(X_val_transformed)
        silhouette = silhouette_score(X_val_transformed, labels)
        avg_silhouette_score += silhouette

    avg_silhouette_score /= k_fold.n_splits
    print("Average silhouette score:", avg_silhouette_score)

    if avg_silhouette_score > best_silhouette_score:
        best_silhouette_score = avg_silhouette_score
        best_ngram_range = ngram_range

print("Best n-gram range:", best_ngram_range)
print("Best silhouette score:", best_silhouette_score)