from sklearn.cluster import KMeans
from vector_store import TfidfStore
import pickle
import os
import joblib
import numpy as np


class KmeanCluster:
    def __init__(self):
        self.cluster_dir = "./cluster_model"
        with open(f"./vector_store/wiki_sg_exclusive_1_1_None_1_50000_True.pkl/tfidf_vectorizer.pkl", "rb") as f:
            self.docs, self.tfidf_vectors = pickle.load(f)
        self.tfidf_vectorizer = joblib.load("./vector_store/wiki_sg_exclusive_1_1_None_1_50000_True.pkl/tfidf_vectorizer.joblib")
    
    def load_model(self) -> KMeans:
        model_path = os.path.join(self.cluster_dir, "kmeans_model.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.fit()
            if not os.path.exists(self.cluster_dir):
                os.mkdir(self.cluster_dir)
            joblib.dump(self.model, f'{self.cluster_dir}/kmeans_model.pkl')
        return self.model
    
    def fit(self, num:int) -> KMeans:
        self.model = KMeans(n_clusters=num, random_state=42)
        self.model.fit(self.tfidf_vectors)
        return self.model
    
    # get popular words for clusters
    def get_cluster_word(self, topk = 5) -> list[str]:
        words = []
        centroids = self.model.cluster_centers_
        for centroid in centroids:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            top_idx = np.argsort(centroid)[-topk:][::-1]
            words.append([feature_names[i] for i in top_idx]) 
        return words

if __name__ == "__main__":
    kmean_cluster = KmeanCluster()
    kmean_cluster.load_model()
    print(kmean_cluster.get_cluster_word())
    
    """
    output:
    [['singapore', 'series', 'art', 'life', 'death'], 
    ['university', 'school', 'students', 'education', 'college'], 
    ['games', 'team', 'tournament', 'world', 'cup'], 
    ['album', 'music', 'song', 'released', 'band'], 
    ['film', 'films', 'festival', 'best', 'directed'], 
    ['singapore', 'government', 'company', 'new', 'chinese'], 
    ['league', 'club', 'season', 'football', 'team'], 
    ['war', 'squadron', 'regiment', 'battalion', 'army'], 
    ['station', 'airport', 'line', 'terminal', 'bus'], 
    ['race', 'lap', 'formula', 'prix', 'season']]
    """    
     