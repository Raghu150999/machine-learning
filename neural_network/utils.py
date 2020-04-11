import numpy as np 

def train_test_split(X, y, split_ratio=0.8, shuffle=False):
    l = len(y)
    n_train = int(l * split_ratio)
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]
    if shuffle:
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        idxs = np.random.randint(0, l, n_train)
        idxs = set(idxs)
        for i in range(l):
            if i in idxs:
                X_train.append(X[i])
                y_train.append(y[i])
            else:
                X_test.append(X[i])
                y_test.append(y[i])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test