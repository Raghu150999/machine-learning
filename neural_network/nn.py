import numpy as np 
import pandas as pd 
from utils import train_test_split
from layer import Layer
from dropout import Dropout
import matplotlib.pyplot as plt

np.random.seed(1234)
eps = np.finfo(np.float32).eps
np.seterr(all='raise')

class NeuralNetwork:

    def __init__(self, filename='housepricedata.csv'):
        '''
        Load data and initialise layers for neural network
        Args:
            filename: path/to/data.csv
        '''
        self.load_data(filename)
        self.layers = []
        self.history_loss = []
        self.history_val_loss = []
    
    def load_data(self, filename):
        df = pd.read_csv(filename, header=None)
        dfx = df.iloc[:, :-1]
        dfx = (dfx - dfx.mean()) / (dfx.max() - dfx.min())
        X = dfx.values
        y = df.iloc[:, -1].values
        self.d = X.shape[1]
        self.out = self.d
        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(X, y)
        self.y_test = self.y_test.reshape(-1, 1)
    
    def add_layer(self, nodes, activation='relu'):
        '''
        add/append layer with 'nodes'
        Args:
            nodes: number of nodes in the layer
            activation: activation function to be used. available: 'relu', 'sigmoid'
        '''
        self.layers.append(Layer(self.out, nodes, activation))
        self.out = nodes
    
    def add_dropout(self, dropout_keep_prob=0.8):
        '''
        add/append dropout layer
        Args:
            dropout_keep_prob: probability that the neuron is not dropped during training
        '''
        self.layers.append(Dropout(dropout_keep_prob))

    def launch_training(self, epochs=1000, learning_rate=0.05, batch_size=64):
        self.train(self.X_train, self.y_train, epochs, learning_rate, batch_size)
        print()
        print('Test')
        accuracy, f1score = self.evaluate()
        print('Accuracy: {:.3f}\nF1-score {:.3f}'.format(accuracy, f1score))
        print('Train')
        accuracy, f1score = self.evaluate(use_training_data=True)
        print('Accuracy: {:.3f}\nF1-score {:.3f}'.format(accuracy, f1score))

    def evaluate(self, use_training_data=False):
        X = self.X_test
        y = self.y_test
        if use_training_data:
            X = self.X_train
            y = self.y_train
        yhat = self.predict(X, training=False)
        yhat = np.round(yhat)
        y = y.reshape(-1, 1)
        tp = np.sum(np.int32(((y == 1) & (yhat == 1))))
        fp = np.sum(np.int32(((y == 0) & (yhat == 1))))
        tn = np.sum(np.int32(((y == 0) & (yhat == 0))))
        fn = np.sum(np.int32(((y == 1) & (yhat == 0))))
        accuracy = (tp + tn) / len(y)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1measure = (2 * precision * recall) / (precision + recall)
        return accuracy, f1measure

    def train(self, X_train, y_train, epochs, learning_rate, batch_size):
        y_train = y_train.reshape(-1, 1)
        for epoch in range(epochs):
            batch = BatchLoader(X_train, y_train, batch_size)
            for i in range(batch.num_batches):
                # get next batch
                X, y = batch.next_batch()

                # Forward propagation
                yhat = self.predict(X)

                # Backward propagation
                delta = self.compute_delta(yhat, y)
                for layer in self.layers[::-1]:
                    delta = layer.backward_propagation(delta, learning_rate)

            # Compute loss
            yhat = self.predict(X_train, training=False)
            loss = self.cross_entropy_loss(yhat, y_train)
            self.history_loss.append(loss)
            print(f'Epoch: {epoch + 1}, loss: {loss}', end='\r')

            # validation loss
            yhat = self.predict(self.X_test, training=False)
            loss = self.cross_entropy_loss(yhat, self.y_test)
            self.history_val_loss.append(loss)

            # Evaluate model at step
            if (epoch + 1) % 500 == 0:
                print()
                accuracy, f1score = self.evaluate()
                print('Accuracy: {:.3f}\nF1-score {:.3f}'.format(accuracy, f1score))

            # TODO: compute test loss for comparison
        
    def predict(self, x, training=True):
        feed = x
        for layer in self.layers:
            feed = layer.forward_propagation(feed, training=training)
        return feed
    
    def compute_delta(self, y, t):
        '''
        Args:
            y: predicted output
            t: actual output
        '''
        delta = -t / (y + eps) + (1 - t) / (1 - y + eps)
        return delta / len(y)
    
    def cross_entropy_loss(self, yhat, y):
        '''
        cross entropy loss function, used for classification problems
        '''
        t = -(y * np.log(yhat + eps) + (1 - y) * np.log((1 - yhat + eps)))
        return np.sum(t) / len(y)


class BatchLoader:

    '''
    Helper class, for generating batches
    '''
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.num_batches = X.shape[0] // batch_size + 1
        self.batch_size = batch_size
    
    def next_batch(self):
        X = self.X[:self.batch_size]
        y = self.y[:self.batch_size]
        self.X = self.X[self.batch_size:]
        self.y = self.y[self.batch_size:]
        return X, y

def logistic_regression():
    '''
    Logistic regression using neural network
    Note: 'sigmoid' activation at output layer (classification problem)
    '''
    # Accuracy: 0.894 F1-score 0.893
    nn = NeuralNetwork()
    nn.add_layer(1, activation='sigmoid')
    nn.launch_training(epochs=10000, learning_rate=0.1, batch_size=32)
    return nn

def neural_net():
    '''
    Neural network architecture
    '''
    # Accuracy: 0.928 F1-score 0.926
    nn = NeuralNetwork()
    nn.add_layer(8)
    nn.add_layer(8)
    nn.add_layer(8)
    nn.add_layer(8)
    nn.add_layer(1, activation='sigmoid')
    nn.launch_training(epochs=1100, learning_rate=0.05, batch_size=64)
    return nn

def neural_net_with_dropout():
    '''
    Neural network architecture
    '''
    # Accuracy: 0.925 F1-score 0.924
    nn = NeuralNetwork()
    nn.add_layer(64)
    nn.add_dropout(0.8)
    nn.add_layer(32)
    nn.add_dropout(0.8)
    nn.add_layer(8)
    nn.add_layer(1, activation='sigmoid')
    nn.launch_training(epochs=1000, learning_rate=0.1, batch_size=64)
    return nn

if __name__ == "__main__":
    # nn = logistic_regression()
    nn = neural_net()
    # nn = neural_net_with_dropout()

    # plot loss curve
    x = np.arange(len(nn.history_loss))
    plt.plot(x, nn.history_loss, label='Train loss')
    plt.plot(x, nn.history_val_loss, label='Validation loss')
    plt.xlabel('No of iterations')
    plt.ylabel('Average cross entropy loss')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig('plots/train_loss.eps', format='eps')

