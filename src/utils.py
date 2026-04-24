import numpy as np


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

    # select informative features
    informative_idx = np.zeros(n_features, dtype=bool)
    informative_features = rng.choice(n_features, n_informative, replace=False)
    informative_idx[informative_features] = True

    X_list = []
    y_list = []
    true_outliers = []

    for k in range(n_clusters):

        # generate cluster mean for informative features
        mu = np.zeros(n_features)

        signs = rng.choice([-1, 1], size=n_informative)
        values = rng.uniform(3, 6, size=n_informative)

        mu[informative_features] = signs * values

        # generate cluster data
        X_k = rng.normal(loc=mu, scale=1.0, size=(n_per_cluster, n_features))
        y_k = np.full(n_per_cluster, k)

        # introduce outliers
        n_out = int(contamination * n_per_cluster)
        out_idx = rng.choice(n_per_cluster, n_out, replace=False)

        # generate outlier shift vector
        b = np.zeros(n_features)
        signs_b = rng.choice([-1, 1], size=n_features)
        values_b = rng.uniform(7, 13, size=n_features)
        b = signs_b * values_b

        # replace outliers with shifted values
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