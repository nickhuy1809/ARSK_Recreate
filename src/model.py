import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def evaluate_result(weights, errors, true_outliers, informative_idx):
    """
    Evaluate the number of outliers and features based on the actual 
    sparsity found by the ARSK algorithm.
    """
    error_norms = np.linalg.norm(errors, axis=1)
    detected_outliers = error_norms > 1e-6 
    
    n_outliers_detected = np.sum(detected_outliers & true_outliers)
    
    detected_features = weights > 1e-6
    
    n_features_detected = np.sum(detected_features[informative_idx])
    
    return n_outliers_detected, n_features_detected

def soft_threshold_group(z: np.ndarray, lam: float) -> np.ndarray:
    """
    Soft-threshold cho vector (group).

    Parameters
    ----------
    z : np.ndarray
        Input vector (shape: p,)
    lam : float
        Regularization parameter λ1

    Returns
    -------
    np.ndarray
        Vector after thresholding
    """
    norm = np.linalg.norm(z)
    if norm == 0:
        return np.zeros_like(z)
    return max(0, 1 - lam / norm) * z

def soft_threshold_scalar(x: float, lam: float) -> float:
    """
    Soft-threshold cho scalar.

    Parameters
    ----------
    x : float
        Input value
    lam : float
        Regularization parameter λ2

    Returns
    -------
    float
        Value after thresholding
    """
    if x > lam:
        return x - lam
    elif x < -lam:
        return x + lam
    return 0.0

def scad_threshold_scalar(x, lam, a=3.7):
    """
    SCAD threshold cho scalar.
    """
    abs_x = abs(x)

    if abs_x <= lam:
        return np.sign(x) * max(abs_x - lam, 0)
    elif abs_x <= a * lam:
        return ((a - 1) * x - np.sign(x) * a * lam) / (a - 2)
    else:
        return x
    
def scad_threshold_group(z, lam, a=3.7):
    """
    SCAD threshold cho vector (group).
    """
    norm = np.linalg.norm(z)

    if norm == 0:
        return np.zeros_like(z)

    # soft part
    def soft(z, lam):
        n = np.linalg.norm(z)
        return max(0, 1 - lam / n) * z

    if norm <= 2 * lam:
        return soft(z, lam)

    elif norm <= a * lam:
        return ((a - 1) * soft(z, lam * a / (a - 1))) / (a - 2)

    else:
        return z

def compute_Q(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Calculate the Q_j values for each feature j based on 
    the current clustering labels.

    Parameters
    ----------
    X : np.ndarray
        Data (n, p)
    labels : np.ndarray
        Cluster labels

    Returns
    -------
    np.ndarray
        Vector Q_j (p,)
    """
    n, p = X.shape
    Q = np.zeros(p)
    overall_mean = np.mean(X, axis=0)

    for j in range(p):
        total = np.sum((X[:, j] - overall_mean[j]) ** 2)
        within = 0.0

        for k in np.unique(labels):
            idx = labels == k
            if np.sum(idx) == 0:
                continue
            mean_k = np.mean(X[idx, j])
            within += np.sum((X[idx, j] - mean_k) ** 2)

        Q[j] = total - within

    return Q

def arsk(
    X,
    n_clusters,
    lambda1,
    lambda2,
    thresh_E="soft",
    thresh_w="soft",
    max_iter=30,
    tol=1e-4,
    random_state=42
):
    """
    ARSK clustering (fixed version - stable)

    Parameters
    ----------
    X : array (n, p)
    n_clusters : int
    lambda1 : float (robustness)
    lambda2 : float (sparsity)
    thresh_E : "soft" or "scad"
    thresh_w : "soft" or "scad"

    Returns
    -------
    dict
    """

    np.random.seed(random_state)
    n, p = X.shape

    w = np.ones(p) / np.sqrt(p)
    E = np.zeros((n, p))

    prev_labels = None

    for _ in range(max_iter):

        # ===== Step 1: weighted data =====
        X_star = (X - E) * w

        if np.allclose(X_star, 0):
            X_star = X.copy()

        # ===== Step 2: clustering =====
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state
        )
        labels = kmeans.fit_predict(X_star)

        if len(np.unique(labels)) < n_clusters:
            break

        # ===== Step 3: update E =====
        E_new = np.zeros_like(E)

        for k in range(n_clusters):
            idx = labels == k
            if np.sum(idx) == 0:
                continue

            mu_k = np.mean(X[idx] - E[idx], axis=0)

            for i in np.where(idx)[0]:
                z = (X[i] - mu_k) * w

                if thresh_E == "soft":
                    E_new[i] = soft_threshold_group(z, lambda1)
                else:
                    E_new[i] = scad_threshold_group(z, lambda1)

        # ===== Step 4: normalize E =====
        for j in range(p):
            if abs(w[j]) > 1e-8:
                E_new[:, j] /= w[j]

        # Clip E to prevent numerical issues
        E_new = np.clip(E_new, -1e3, 1e3)

        # ===== Step 5: update w =====
        X_prime = X - E_new
        Q = compute_Q(X_prime, labels)

        S = []
        for q in Q:
            if thresh_w == "soft":
                S.append(soft_threshold_scalar(q, lambda2))
            else:
                S.append(scad_threshold_scalar(q, lambda2))

        S = np.array(S)

        if np.linalg.norm(S) < 1e-8:
            w_new = np.ones(p) / np.sqrt(p)
        else:
            w_new = S / np.linalg.norm(S)

        # ===== convergence =====
        if prev_labels is not None:
            if np.array_equal(labels, prev_labels):
                break

        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new
        E = E_new
        prev_labels = labels

    return {
        "labels": labels,
        "weights": w,
        "errors": E
    }

def compute_DR(X, labels, weights, errors):
    """
    Compute robust between-cluster sum of squares (D_R).
    """
    X_prime = X - errors
    n, p = X.shape
    overall_mean = np.mean(X_prime, axis=0)

    total = 0
    within = 0

    for j in range(p):
        total += weights[j] * np.sum((X_prime[:, j] - overall_mean[j])**2)

        for k in np.unique(labels):
            idx = labels == k
            if np.sum(idx) == 0:
                continue
            mean_k = np.mean(X_prime[idx, j])
            within += weights[j] * np.sum((X_prime[idx, j] - mean_k)**2)

    return total - within

def permute_dataset(X, random_state=42):
    rng = np.random.default_rng(random_state)
    X_perm = X.copy()

    for j in range(X.shape[1]):
        rng.shuffle(X_perm[:, j])

    return X_perm

def compute_gap(X, lambda1, lambda2, thresh_E="soft", thresh_w="soft", B=25, random_state=42):
    rng = np.random.default_rng(random_state)

    # Run ARSK on the original dataset with the specified thresholding configuration
    res = arsk(
        X,
        n_clusters=3,
        lambda1=lambda1,
        lambda2=lambda2,
        thresh_E=thresh_E, 
        thresh_w=thresh_w, 
        random_state=rng.integers(1e9)
    )

    DR = compute_DR(X, res["labels"], res["weights"], res["errors"])
    log_perm = []

    for _ in range(B):
        Xb = permute_dataset(X, rng.integers(1e9))
        # Run ARSK on the permuted dataset with the same thresholding configuration
        res_b = arsk(
            Xb,
            n_clusters=3,
            lambda1=lambda1,
            lambda2=lambda2,
            thresh_E=thresh_E, 
            thresh_w=thresh_w,
            random_state=rng.integers(1e9)
        )
        DR_b = compute_DR(Xb, res_b["labels"], res_b["weights"], res_b["errors"])
        log_perm.append(np.log(DR_b + 1e-10))

    gap = np.log(DR + 1e-10) - np.mean(log_perm)
    return gap

def select_lambda(X, thresh_E="soft", thresh_w="soft"):
    """
    Select optimal lambda1 and lambda2 by maximizing the Gap statistic.
    """
    # Range of lambda values to search
    lambda1_values = np.linspace(0.5, 5, 5) 
    lambda2_values = np.linspace(0.5, 5, 5)

    lambda1_fixed = 2.0
    best_lambda2 = None
    best_gap = -np.inf

    for l2 in lambda2_values:
        gap = compute_gap(X, lambda1_fixed, l2, thresh_E=thresh_E, thresh_w=thresh_w)
        if gap > best_gap:
            best_gap = gap
            best_lambda2 = l2

    best_lambda1 = None
    best_gap = -np.inf

    for l1 in lambda1_values:
        gap = compute_gap(X, l1, best_lambda2, thresh_E=thresh_E, thresh_w=thresh_w)
        if gap > best_gap:
            best_gap = gap
            best_lambda1 = l1

    return best_lambda1, best_lambda2