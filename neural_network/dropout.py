import numpy as np
from layer import Layer

np.random.seed(1234)

class Dropout:

    '''
    Layer class abstraction for neural network layer
    '''
    def __init__(self, dropout_keep_prob=0.8):
        '''
        Initialize a dropout layer with provided dropout_keep_prob
        '''
        self.dropout_keep_prob = dropout_keep_prob
        assert(dropout_keep_prob > 0 and dropout_keep_prob <= 1)
    
    def forward_propagation(self, x, training=True):
        '''
        Perform feed forward step, with dropout
        Args:
            x: input shape: [batch_size, in_nodes]
        Returns:
            z: output after forward prop. shape: [batch_size, out_nodes]
        '''
        # use dropout during training
        if training:
            p = self.dropout_keep_prob
            self.b = np.random.binomial(1, p, size=x.shape) * (1 / p)
        else:
            p = 1.0
            self.b = np.ones(x.shape)
        self.x = x
        z = self.b * self.x
        return z
    
    def backward_propagation(self, delta, learning_rate):
        '''
        Perform backward propagation step
        Args:
            delta: error from previous layer. shape: [batch_size, out_nodes]
        '''
        # compute error for previous layer
        delta = delta * self.b
        return delta