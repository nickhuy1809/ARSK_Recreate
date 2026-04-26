import numpy as np
from scipy.special import comb

def compute_cer(y_true, labels_pred, errors, true_outliers, n_clusters):
    """
    Tính CER dựa trên việc so sánh các cặp (i, j).
    Outliers được gán vào cụm thứ K (index từ 0 đến K).
    """
    n = len(y_true)
    
    # Nhận diện outlier dự đoán (norm > 1e-6)
    is_outlier_pred = np.linalg.norm(errors, axis=1) > 1e-6
    
    # Gán nhãn cho outliers là cụm n_clusters (nhóm K+1)
    y_pred_final = labels_pred.copy()
    y_pred_final[is_outlier_pred] = n_clusters
    
    y_true_final = y_true.copy()
    y_true_final[true_outliers] = n_clusters

    # Tính số cặp (i, j) bị phân loại sai
    wrong_pairs = 0
    # Tối ưu hóa: chỉ duyệt nửa ma trận cặp
    for i in range(n):
        for j in range(i + 1, n):
            same_true = (y_true_final[i] == y_true_final[j])
            same_pred = (y_pred_final[i] == y_pred_final[j])
            if same_true != same_pred:
                wrong_pairs += 1
                
    return wrong_pairs / comb(n, 2)