import numpy as np

np.random.seed(1234)

def relu(x):
    '''
    ReLU: rectified linear unit activation function
    '''
    return np.maximum(x, 0, x)

def relud(y):
    '''
    differentiation of relu activation
    '''
    return np.float32(y > 0)

def sigmoid(x):
    '''
    sigmoid activation function
    '''
    return 1 / (1 + np.exp(-x))

def sigmoidd(y):
    '''
    differentiation of sigmoid function
    '''
    return y * (1 - y)


class Layer:

    '''
    Layer class abstraction for neural network layer
    '''
    def __init__(self, in_nodes, out_nodes, activation):
        '''
        Initialize shape and weights of layer
        Args:
            in_nodes: number of nodes at the input
            out_nodes: number of nodes at the output
            activation: activation function to be used
        '''
        self.in_nodes = in_nodes 
        self.out_nodes = out_nodes
        func = {
            'relu': relu,
            'sigmoid': sigmoid
        }
        funcd = {
            'relu': relud,
            'sigmoid': sigmoidd
        }

        # get activation function
        self.activation = func[activation]
        self.actd = funcd[activation]

        # initialise weights and bias
        self.w = np.random.normal(size=(in_nodes, out_nodes))
        self.b = np.zeros(out_nodes)
    
    def forward_propagation(self, x):
        '''
        Perform feed forward step
        Args:
            x: input shape: [batch_size, in_nodes]
        Returns:
            z: output after forward prop. shape: [batch_size, out_nodes]
        '''
        self.x = x
        a = np.matmul(self.x, self.w) + self.b
        self.z = self.activation(a)
        return self.z
    
    def backward_propagation(self, delta, learning_rate):
        '''
        Perform backward propagation step
        Args:
            delta: error from previous layer. shape: [batch_size, out_nodes]
        '''
        # compute error for previous layer
        g = delta * self.actd(self.z)
        delta = np.matmul(g, self.w.T)

        # compute gradients
        grad = np.matmul(self.x.T, g)

        # update weights
        self.w = self.w - learning_rate * grad
        return delta