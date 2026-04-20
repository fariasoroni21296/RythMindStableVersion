from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def evaluate(X, labels, true_labels=None):
    result = {}

    if len(set(labels)) > 1:
        result["silhouette"] = silhouette_score(X, labels)
        result["db_index"] = davies_bouldin_score(X, labels)
    else:
        result["silhouette"] = -1
        result["db_index"] = -1

    if true_labels is not None:
        result["ARI"] = adjusted_rand_score(true_labels, labels)

    return result