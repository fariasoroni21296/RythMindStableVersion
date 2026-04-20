from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def run_kmeans(X, k=10):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    return labels, model

def run_agglomerative(X, k=10):
    model = AgglomerativeClustering(n_clusters=k)
    return model.fit_predict(X)

def run_dbscan(X, eps=0.5):
    model = DBSCAN(eps=eps)
    return model.fit_predict(X)