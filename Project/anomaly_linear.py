from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def anomaly_detection_PCA(X: np.array):
    '''
    X : not scaled data
    '''
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA()
    pca_result = pca.fit_transform(X)
    anomaly_score = np.sum(np.abs(pca_result) / pca.explained_variance_, axis=1)
    return anomaly_score