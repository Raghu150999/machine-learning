import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse


class FisherDiscriminant:

    def normal_dist(self, x, mu, sigma):
        t = np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))
        t = t / (sigma * math.sqrt(2 * math.pi))
        return t

    def evaluate(self, df_test):
        X = df_test.iloc[:, :-1].values
        y = df_test.iloc[:, -1].values
        # Project
        x = np.matmul(self.w, X.T)
        b = self._m0 > self.thres
        res = (x > self.thres) ^ b
        yhat = np.int32(res)
        tp = np.sum(np.int32(((y == 1) & (yhat == 1))))
        fp = np.sum(np.int32(((y == 0) & (yhat == 1))))
        tn = np.sum(np.int32(((y == 0) & (yhat == 0))))
        fn = np.sum(np.int32(((y == 1) & (yhat == 0))))
        accuracy = (tp + tn) / len(y)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1measure = (2 * precision * recall) / (precision + recall)
        return accuracy, f1measure

    def __init__(self, data_dir='a1_data/a1_d1.csv'):

        df = pd.read_csv(data_dir, header=None)

        # Split train and test parts
        l = len(df)
        n_train = int(l * 0.8)
        df_test = df.iloc[n_train:, :]
        df = df.iloc[:, :n_train]

        df0 = df[df.iloc[:, -1] == 0]
        df1 = df[df.iloc[:, -1] == 1]

        X0 = df0.iloc[:, :-1].values
        X1 = df1.iloc[:, :-1].values

        # Calculate means
        m0 = np.mean(X0, axis=0)
        m1 = np.mean(X1, axis=0)

        t0 = (X0 - m0)
        t1 = (X1 - m1)

        # Calculate Sw in between covariance
        Sw = np.matmul(t0.T, t0) + np.matmul(t1.T, t1)

        # Sw inverse
        Sw_1 = np.linalg.inv(Sw)
        diff = (m1 - m0)
        w = np.matmul(Sw_1, diff.T)

        # Calculate norm
        norm = np.sum(w ** 2) ** 0.5

        # Projection vector
        w = w / norm

        self.w = w

        # Project to 1D
        x0 = np.matmul(w, X0.T)
        x1 = np.matmul(w, X1.T)

        # Compute means in 1D
        _m0 = np.mean(x0)
        _m1 = np.mean(x1)
        self._m0 = _m0
        self._m1 = _m1

        d0 = (x0 - _m0) ** 2
        d1 = (x1 - _m1) ** 2

        # Compute stdevs
        s0 = math.sqrt(np.mean(d0))
        s1 = math.sqrt(np.mean(d1))

        # Compute threshold by intersection
        mn = min(_m0, _m1)
        mx = max(_m0, _m1)
        x = np.arange(start=mn, stop=mx, step=0.001)
        y0 = self.normal_dist(x, _m0, s0)
        y1 = self.normal_dist(x, _m1, s1)
        diff = np.abs(y1 - y0)
        idx = np.argmin(diff)

        thres = x[idx]
        self.thres = thres

        # Plot normal distribution
        _x0 = np.arange(start=-1.5+_m0, stop=1.5+_m0, step=0.01)
        _y0 = self.normal_dist(_x0, _m0, s0)
        plt.plot(_x0, _y0, color='blue', label='class 0')

        _x1 = np.arange(start=-1.5+_m1, stop=1.5+_m1, step=0.01)
        _y1 = self.normal_dist(_x1, _m1, s1)
        plt.plot(_x1, _y1, color='red', label='class 1')
        plt.annotate('threshold={:.3f}'.format(self.thres), xy=(x[idx], y0[idx]), xytext=(
            x[idx], y0[idx]+.2), arrowprops=(dict(facecolor='black', shrink=0.005)))
        plt.title('Normal Distribution plot')
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.legend()
        plt.show()
        # plt.savefig('normal_dist.eps', format='eps')

        # Evaluate
        accuracy, f1_score = self.evaluate(df_test)
        print(f'Accuracy: {accuracy}, F1 Score: {f1_score}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fisher Discriminant')
    parser.add_argument('--data_dir', type=str, default='a1_data/a1_d1.csv')
    args = parser.parse_args()
    fd = FisherDiscriminant(args.data_dir)

