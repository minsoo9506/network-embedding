import numpy as np

class MF:
    def __init__(self, embedding_dim=4, learning_rate=0.005, n_iter=200, adj_matrix=None):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.adj_matrix = adj_matrix
        self.embedding_vec = np.random.rand(len(adj_matrix),embedding_dim)
        
    def train(self):
        for _ in range(self.n_iter):
            e_matrix = np.matmul(self.embedding_vec, self.embedding_vec.T) - self.adj_matrix
            for i in range(0, len(self.adj_matrix)):
                for j in range(0, len(self.adj_matrix)):
                    if i == j:
                        continue
                    self.embedding_vec[i] -= self.learning_rate * e_matrix[i][j] * self.embedding_vec[j]

    def show_embedding(self):
        return self.embedding_vec


