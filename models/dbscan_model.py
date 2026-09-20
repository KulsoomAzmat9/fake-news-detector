import pandas as pd
from sklearn.cluster import DBSCAN

def run_dbscan(X):
    print("\n=== UNSUPERVISED: DBSCAN ===")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X.toarray())
    
    cluster_counts = pd.Series(clusters).value_counts().to_dict()
    noise_points = (clusters == -1).sum()
    
    print(f"DBSCAN Cluster Distribution: {cluster_counts}")
    print(f"Noise points found: {noise_points}")