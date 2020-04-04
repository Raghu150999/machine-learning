import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer 

# required nltk wordnet (Download when running for first time)
# nltk.download('wordnet')

class NaiveBayes:

    def __init__(self):
        '''
        Initialize and load dataset
        '''
        # Get all stop words
        self.stop_words = set(stopwords.words('english'))

        # Lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Load data
        self.load_data()

        X_train, y_train, X_test, y_test = self.train_test_split(self.X, self.y)

        # Train model
        self.train(X_train, y_train)

        # Evaluate
        accuracy, f1measure = self.evaulate(X_test, y_test)

        print('Accuracy: {:.3f}, F1-score: {:.3f}'.format(accuracy,f1measure))


    def test_model(self):
        '''
        Helper for stress testing model
        '''
        X_train, y_train, X_test, y_test = self.train_test_split(self.X, self.y, shuffle=True)

        # Train model
        self.train(X_train, y_train)

        # Evaluate
        accuracy, f1measure = self.evaulate(X_test, y_test)

        print('Accuracy: {:.3f}, F1-score: {:.3f}'.format(accuracy, f1measure))
        return accuracy, f1measure

    def train(self, X, y):
        # Get prior probabilities
        l = len(y)
        y = np.array(y)
        ones = np.sum(y)
        assert(0 < ones and ones < l)
        self.p = [(l - ones)/l, ones/l]

        vocab_len = len(self.vocab)
        # Initialize with non zero probabilites (smoothing)
        prob = {}
        for word in self.vocab:
            prob[(word, 0)] = 1
            prob[(word, 1)] = 1
        
        for i, text in enumerate(X):
            for word in text:
                prob[(word, y[i])] += 1
        
        # Compute probablities with smoothing
        for word in self.vocab:
            prob[(word, 0)] = prob[(word, 0)] / (l - ones + 1)
            prob[(word, 1)] = prob[(word, 1)] / (ones + 1)
        
        self.prob = prob

    def test(self, processed_text):
        # processed_text = self.preprocessor(text)
        # Load prior probablities
        p = [self.p[0], self.p[1]]
        p = np.log(p)
        for word in processed_text:
            p[0] += np.log(self.prob[(word, 0)])
            p[1] += np.log(self.prob[(word, 1)])
        output = int(p[1] > p[0])
        return output

    def evaulate(self, X, y):
        yhat = []
        for text in X:
            yhat.append(self.test(text))
        yhat = np.array(yhat, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        tp = np.sum(np.int32(((y == 1) & (yhat == 1))))
        fp = np.sum(np.int32(((y == 0) & (yhat == 1))))
        tn = np.sum(np.int32(((y == 0) & (yhat == 0))))
        fn = np.sum(np.int32(((y == 1) & (yhat == 0))))
        accuracy = (tp + tn) / len(y)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1measure = (2 * precision * recall) / (precision + recall)
        return accuracy, f1measure

    def load_data(self, file='a1_data/a1_d3.txt'):
        '''
        Loads vocabulary and dataset
        Args:
            file: file path to data.txt
        '''
        # Vocabulary
        self.vocab = set()
        self.X = []
        self.y = []
        with open(file, 'r') as f:
            for line in f:
                l = line.strip()
                target = int(l[-1])
                text = l[:-1]
                processed_text = self.preprocessor(text)
                for token in processed_text:
                    self.vocab.add(token)
                self.X.append(processed_text)
                self.y.append(target)

    def preprocessor(self, text):
        """
        Performs preprocessing operations: tokenize, lowercase, remove stopwords, lemmatize
        Args:
            text: raw text
        """
        # Remove punctuations
        tokens = self.tokenizer.tokenize(text)

        # Convert to lower case
        for i, token in enumerate(tokens):
            tokens[i] = token.lower()
        
        stop_words = set([
            'i', 'the', 'is', 'a'
        ])
        # Removing stop words
        tokens = [w for w in tokens if not w in stop_words]

        # Perform lemmatization
        for i, token in enumerate(tokens):
            tokens[i] = self.lemmatizer.lemmatize(token)
        return tokens

    def train_test_split(self, X, y, split_ratio=0.8, shuffle=False):
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
        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    nb = NaiveBayes()
    avg_accuracy = 0.
    avg_f1score = 0.
    for i in range(5):
        accuracy, f1measure = nb.test_model()
        avg_accuracy += accuracy
        avg_f1score += f1measure
    avg_accuracy /= 5
    avg_f1score /= 5
    print('Average accuracy {:.3f}, Average F1-score {:.3f}'.format(avg_accuracy, avg_f1score))