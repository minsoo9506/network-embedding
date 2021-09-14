from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, precision_score, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def make_graph(X, cos_sim_threshold):
    sim = cosine_similarity(X, X)
    for idx in range(0, len(X)):
        sim[idx,idx] = 0
    adj_mat = np.where(sim > cos_sim_threshold, 1, 0)
    graph = nx.from_numpy_matrix(adj_mat)
    return graph

def scoring(true, pred):
    confusion = confusion_matrix(true, pred)
    con_df = pd.DataFrame(confusion, index=['Normal', 'Abnormal'], columns=['Normal', 'Abnormal'])
    
    score_df = pd.DataFrame(index = ['Accuracy','ROC AUC', 'Recall', 'Precision', 'f1 score'], columns=['Score'])
    score_df.iloc[0, :] = round(accuracy_score(true, pred), 3)
    score_df.iloc[1, :] = round(roc_auc_score(true, pred), 3)
    score_df.iloc[2, :] = round(recall_score(true, pred), 3)
    score_df.iloc[3, :] = round(precision_score(true, pred), 3)
    score_df.iloc[4, :] = round(f1_score(true, pred), 3)
    
    print(f'{score_df}\n')
    print(f'confusion matrix \n{con_df}')