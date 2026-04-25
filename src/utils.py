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

def generate_data_sim2(
    n_per_cluster=50,
    n_clusters=3,
    p=50,
    q=5,
    contamination=0.1,
    correlated=False,
    random_state=42
):
    """
    Sinh dữ liệu cho Simulation 2 (Table 2 & 3 trong bài báo).
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Khởi tạo ma trận hiệp phương sai Sigma
    if not correlated:
        # Trường hợp biến độc lập (Table 2)
        sigma = np.eye(p)
    else:
        # Trường hợp biến có tương quan (Table 3) theo công thức Σ = QPQ^T
        # Q: Ma trận xoay ngẫu nhiên (orthogonal matrix)
        Q = ortho_group.rvs(dim=p, random_state=random_state)
        # P: Ma trận đường chéo với rho_t ~ U(0.1, 1)
        rho = rng.uniform(0.1, 1, size=p)
        P = np.diag(rho)
        sigma = Q @ P @ Q.T

    # 2. Xác định các biến thông tin (q biến)
    informative_idx = np.zeros(p, dtype=bool)
    informative_features = rng.choice(p, q, replace=False)
    informative_idx[informative_features] = True

    X_list, y_list, true_outliers = [], [], []

    for k in range(n_clusters):
        # Sinh vector trung bình mu_k
        mu = np.zeros(p)
        # mu_j ~ U(3, 6) hoặc U(-6, -3) cho các biến thông tin
        signs = rng.choice([-1, 1], size=q)
        values = rng.uniform(3, 6, size=q)
        mu[informative_features] = signs * values

        # Sinh mẫu bình thường từ phân phối chuẩn đa biến N(mu, sigma)
        X_k = rng.multivariate_normal(mean=mu, cov=sigma, size=n_per_cluster)
        y_k = np.full(n_per_cluster, k)

        # Chọn ngẫu nhiên các điểm làm Outliers
        n_out = int(contamination * n_per_cluster)
        out_idx = rng.choice(n_per_cluster, n_out, replace=False)

        # Sinh độ chệch b_j ~ U(7, 13) hoặc U(-13, -7)
        b = rng.choice([-1, 1], size=p) * rng.uniform(7, 13, size=p)
        
        # Thay thế các mẫu bằng outlier: X_k = N(mu + b, sigma)
        X_k[out_idx] = rng.multivariate_normal(mean=mu + b, cov=sigma, size=n_out)

        true_out = np.zeros(n_per_cluster, dtype=bool)
        true_out[out_idx] = True

        X_list.append(X_k)
        y_list.append(y_k)
        true_outliers.append(true_out)

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(true_outliers), informative_idx