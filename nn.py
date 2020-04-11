import numpy as np 
import pandas as pd 
from utils import train_test_split

np.random.seed(1234)

def relu(x):
    return np.maximum(x, 0, x)

def relud(y):
    return np.float32(y > 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidd(y):
    return y * (1 - y)


class Layer:

    def __init__(self, in_nodes, out_nodes, activation):
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
        self.activation = func[activation]
        self.actd = funcd[activation]
        self.w = np.random.normal(size=(in_nodes, out_nodes))
        self.b = np.zeros(out_nodes)
    
    def forward_propagation(self, x):
        self.x = x
        a = np.matmul(self.x, self.w) + self.b
        self.z = self.activation(a)
        return self.z
    
    def backward_propagation(self, delta, learning_rate):
        g = delta * self.actd(self.z)
        delta = np.matmul(g, self.w.T)
        grad = np.matmul(self.x.T, g)
        self.w = self.w - learning_rate * grad
        return delta

class NeuralNetwork:

    def __init__(self, filename='housepricedata.csv'):
        self.load_data(filename)
        self.layers = []
    
    def load_data(self, filename):
        df = pd.read_csv(filename, header=None)
        dfx = df.iloc[:, :-1]
        dfx = (dfx - dfx.mean()) / (dfx.max() - dfx.min())
        X = dfx.values
        y = df.iloc[:, -1].values
        self.d = X.shape[1]
        self.out = self.d
        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(X, y)
    
    def add_layer(self, nodes, activation='relu'):
        self.layers.append(Layer(self.out, nodes, activation))
        self.out = nodes
    
    def launch_training(self, epochs=1000, learning_rate=0.05, batch_size=64):
        self.train(self.X_train, self.y_train, epochs, learning_rate, batch_size)
        print()
        accuracy, f1score = self.evaluate()
        print('Accuracy: {:.3f}\nF1-score {:.3f}'.format(accuracy, f1score))

    def evaluate(self):
        yhat = self.predict(self.X_test)
        yhat = np.round(yhat)
        y = self.y_test.reshape(-1, 1)
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
                X, y = batch.next_batch()

                # Forward propagation step
                yhat = self.predict(X)

                # Backward propagation step
                delta = self.compute_delta(yhat, y)
                for layer in self.layers[::-1]:
                    delta = layer.backward_propagation(delta, learning_rate)
            # Compute loss
            yhat = self.predict(X_train)
            loss = self.cross_entropy_loss(yhat, y_train)
            print(f'Epoch: {epoch + 1}, loss: {loss}', end='\r')
        
    def predict(self, x):
        feed = x
        for layer in self.layers:
            feed = layer.forward_propagation(feed)
        return feed
    
    def compute_delta(self, y, t):
        '''
        Args:
            y: predicted output
            t: actual output
        '''
        delta = -t / y + (1 - t) / (1 - y)
        return delta / len(y)
    
    def cross_entropy_loss(self, yhat, y):
        t = -(y * np.log(yhat) + (1 - y) * np.log((1 - yhat)))
        return np.sum(t) / len(y)


class BatchLoader:

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
    nn = NeuralNetwork()
    nn.add_layer(1, activation='sigmoid')
    nn.launch_training(epochs=10000, learning_rate=0.1, batch_size=32)

def neural_net():
    nn = NeuralNetwork()
    nn.add_layer(8)
    nn.add_layer(8)
    nn.add_layer(8)
    nn.add_layer(8)
    nn.add_layer(1, activation='sigmoid')
    nn.launch_training(epochs=4000, learning_rate=0.005, batch_size=64)

if __name__ == "__main__":
    # logistic_regression()
    neural_net()


