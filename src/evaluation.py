from sklearn.metrics import silhouette_score, calinski_harabasz_score


def evaluate(X, labels):
    """
    Calculate clustering metrics
    """
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return sil, ch


def evaluate_and_print(X, labels, name="Model"):
    sil, ch = evaluate(X, labels)
    print(f"{name} -> Silhouette: {sil:.3f}, CH Score: {ch:.2f}")
    return sil, ch