import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from utils import train_test_split
import argparse

def sigmoid(x):
    return (1. / (1 + np.exp(-x)))

class LogisticRegression:

    def __init__(self, epochs, learning_rate, reg_type, reg_lambda, filename='data_banknote_authentication.csv'):
        self.load_data(filename)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda

    def load_data(self, filename):
        df = pd.read_csv(filename, header=None)
        dfx = df.iloc[:, :-1]
        dfx.iloc[:, -1] = 1
        X = dfx.values
        y = df.iloc[:, -1].values
        self.d = X.shape[1]
        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(X, y)

    def launch_training(self):
        self.train(self.X_train, self.y_train, self.epochs, self.learning_rate, self.reg_type, self.reg_lambda)
    
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

    def predict(self, X):
        t = np.matmul(X, self.w)
        return sigmoid(t)

    def train(self, X, y, epochs, learning_rate, reg_type, reg_lambda):
        # Initialise weights
        self.w = np.zeros((self.d, 1))
        y = y.reshape(-1, 1)

        for i in range(epochs):
            # predict
            yhat = self.predict(X)

            # compute gradient
            g = self.compute_gradient(X, y, yhat)
            g = g.reshape(-1, 1)

            # compute regularisation
            if reg_type == 'l2':
                r = 2 * reg_lambda * self.w
            elif reg_type == 'l1':
                v = np.int32((self.w > 0))
                v = 2 * v - 1
                r = reg_lambda * v
            else:
                r = 0
            
            # add regularisation
            g = g + r
            g = g.reshape(-1, 1)

            # compute loss
            loss = self.cross_entropy_loss(y, yhat) + np.sum(r)
            print(f'Epoch {i+1}, loss: {loss}', end='\r')

            # update weights
            self.w  = self.w - learning_rate * g
        print()

    def cross_entropy_loss(self, y, yhat):
        t = -(y * np.log(yhat) + (1 - y) * np.log((1 - yhat)))
        return np.sum(t)

    def compute_gradient(self, X, y, yhat):
        t = yhat - y
        t = t.reshape(-1, 1)
        prod = t * X
        return np.sum(prod, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Logistic Regression')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--reg_type', type=str, default='none', help='specify regularisation type, available: l1, l2, none')
    parser.add_argument('--reg_lambda', type=float, default=0.001, help='regularisation lambda')
    args = parser.parse_args()

    lr = LogisticRegression(epochs=args.epochs, learning_rate=args.learning_rate, reg_type=args.reg_type, reg_lambda=args.reg_lambda)
    lr.launch_training()
    accuracy, f1score = lr.evaluate()
    print('Accuracy: {:.3f}\nF1-score {:.3f}'.format(accuracy, f1score))


