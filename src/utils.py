import numpy as np
from scipy.stats import ortho_group

def generate_data(
    n_per_cluster=50,
    n_clusters=3,
    n_features=50,
    n_informative=5,
    contamination=0.1,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    # n_samples = n_per_cluster * n_clusters

    # === chọn informative features ===
    informative_idx = np.zeros(n_features, dtype=bool)
    informative_features = rng.choice(n_features, n_informative, replace=False)
    informative_idx[informative_features] = True

    X_list = []
    y_list = []
    true_outliers = []

    for k in range(n_clusters):

        # === sinh μ_k ===
        mu = np.zeros(n_features)

        signs = rng.choice([-1, 1], size=n_informative)
        values = rng.uniform(3, 6, size=n_informative)

        mu[informative_features] = signs * values

        # === normal samples ===
        X_k = rng.normal(loc=mu, scale=1.0, size=(n_per_cluster, n_features))
        y_k = np.full(n_per_cluster, k)

        # === chọn outliers ===
        n_out = int(contamination * n_per_cluster)
        out_idx = rng.choice(n_per_cluster, n_out, replace=False)

        # === sinh b_j ===
        b = np.zeros(n_features)
        signs_b = rng.choice([-1, 1], size=n_features)
        values_b = rng.uniform(7, 13, size=n_features)
        b = signs_b * values_b

        # === replace bằng outlier distribution ===
        X_k[out_idx] = rng.normal(loc=mu + b, scale=1.0, size=(n_out, n_features))

        true_out = np.zeros(n_per_cluster, dtype=bool)
        true_out[out_idx] = True

        X_list.append(X_k)
        y_list.append(y_k)
        true_outliers.append(true_out)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    true_outliers = np.concatenate(true_outliers)

    return X, y, true_outliers, informative_idx