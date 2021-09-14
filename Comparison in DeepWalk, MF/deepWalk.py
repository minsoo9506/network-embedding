import numpy as np
from typing import List

class DeepWalk:
    def __init__(self,
                 adj_matrix,
                 embedding_dim=2,
                 walks_per_vertex=5,
                 walk_len=10,
                 window_size=3,
                 learning_rate=0.02):
        
        self.adj_matrix = adj_matrix
        self.embedding_dim = embedding_dim
        self.walks_per_vertex = walks_per_vertex
        self.walk_len = walk_len
        self.window_size = window_size
        self.learning_rate = learning_rate
        
        self.w1 = np.random.rand(len(adj_matrix),embedding_dim)
        self.w2 = np.random.rand(embedding_dim,len(adj_matrix))
        
        self.loss = 0.0
        self.epoch_loss = []
    
    def _random_walk(self, start_node: int)-> List:
        walk = [0] * self.walk_len
        walk[0] = start_node
        node = start_node
        for i in range(1, self.walk_len):
            next_node = np.random.choice(np.where(self.adj_matrix[node]==1)[0])
            walk[i] = next_node
            node = next_node
        return walk

    def _softmax(self, a: np.array)-> np.array : 
        c = np.max(a) 
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def _skip_gram_train(self, walk: List)-> None:
        for idx, input_node in enumerate(walk):
            # make dataset
            left_idx = idx - self.window_size
            right_idx = idx + self.window_size
            if left_idx < 0: left_idx = 0
            if right_idx > self.walk_len-1: right_idx = self.walk_len
            left_node = walk[left_idx:idx]
            right_node = walk[idx+1:right_idx+1]
            output_node = left_node + right_node

            # forward
            hidden = self.w1[input_node]
            ## |hidden| = (2,)
            out = np.matmul(hidden, self.w2)
            ## |out| = (34,)

            # loss calculate
            self.loss += (-np.sum(out[output_node]) \
                                   + len(output_node)*np.log(np.sum(np.exp(out))))\
                                / (self.walk_len*self.walks_per_vertex*len(self.adj_matrix)) 

            # backprop and optimize
            dEdo = self._softmax(out) * len(output_node)
            dEdo[output_node] = dEdo[output_node] - 1.0 - self._softmax(out)[output_node]
            dEdw2 = hidden.reshape(self.embedding_dim,1) @ dEdo.reshape(1,len(self.adj_matrix))
            self.w2 = self.w2 - self.learning_rate * dEdw2
            self.w1[input_node] = self.w1[input_node] - \
                self.learning_rate * np.matmul(self.w2, dEdo)

    def train(self)-> float:
        V = np.arange(0, len(self.adj_matrix))
        for _ in range(self.walks_per_vertex):
            # shuffle vertex
            np.random.shuffle(V)
            for start_node in V:
                # random walk
                W = self._random_walk(start_node)
                # skip-gram
                self._skip_gram_train(W)
             # consider epoch as if all node be start_node
             # = consider epoch as walks_per_vertex
            self.epoch_loss.append(self.loss)
            self.loss = 0.0
        return self.epoch_loss
    
    def show_embedding(self):
        return self.w1, self.w2