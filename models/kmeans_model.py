import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(X):
    print("\n=== UNSUPERVISED: K-MEANS ===")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X.toarray())
    print(f"K-Means found 2 clusters. Distribution: {pd.Series(clusters).value_counts().to_dict()}")