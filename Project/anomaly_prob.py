import numpy as np

def anomaly_detection_Malhalanobis(X: np.array):
    X_mean = np.mean(X, axis=0)
    S_inv = np.linalg.inv(np.cov(X, rowvar=False))
    anomaly_score = np.zeros((len(X),))
    for i in range(len(X)):
        anomaly_score[i] = (X[i,:] - X_mean) @ S_inv @ np.transpose(X[i,:] - X_mean)
    anomaly_score = np.sqrt(anomaly_score)
    
    q1 = np.quantile(anomaly_score, 0.25)
    q3 = np.quantile(anomaly_score, 0.75)
    IQR = q3 - q1
    upper = q3 + 1.5*IQR
    result = np.where(anomaly_score < upper, 0, 1)
    
    return anomaly_score, result