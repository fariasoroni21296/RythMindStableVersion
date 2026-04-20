from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def run_kmeans(X, k=5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

def pca_transform(X, n_components=16):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

