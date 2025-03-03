import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.corpus import PlaintextCorpusReader, stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from utils import train_test_split

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

        # stemmer
        self.stemmer = PorterStemmer()
        
        # Tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Load data
        self.load_data()

        X_train, y_train, X_test, y_test = train_test_split(self.X, self.y)

        # Train model
        self.train(X_train, y_train)

        # Evaluate
        accuracy, f1measure = self.evaulate(X_test, y_test)

        print('Accuracy: {:.3f}, F1-score: {:.3f}'.format(accuracy,f1measure))


    def test_model(self):
        '''
        Helper for stress testing model
        '''
        X_train, y_train, X_test, y_test = train_test_split(self.X, self.y, shuffle=True)

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
        # tokens = [w for w in tokens if not w in stop_words]

        # Perform lemmatization
        # for i, token in enumerate(tokens):
        #     tokens[i] = self.lemmatizer.lemmatize(token)

        # Perform stemming
        # for i, token in enumerate(tokens):
        #     tokens[i] = self.stemmer.stem(token)
        return tokens


if __name__ == "__main__":
    nb = NaiveBayes()
    avg_accuracy = 0.
    avg_f1score = 0.
    acc = []
    f1 = []
    for i in range(5):
        accuracy, f1measure = nb.test_model()
        avg_accuracy += accuracy
        avg_f1score += f1measure
        acc.append(accuracy)
        f1.append(f1measure)
    avg_accuracy /= 5
    avg_f1score /= 5
    print('Average accuracy {:.3f}, Average F1-score {:.3f}'.format(avg_accuracy, avg_f1score))
    vacc = 0
    vf1 = 0
    for i in range(len(acc)):
        vacc += (acc[i] - avg_accuracy) ** 2
        vf1 += (f1[i] - avg_f1score) ** 2
    vacc /= 5
    vf1 /= 5
    vacc = vacc ** 0.5
    vf1 = vf1 ** 0.5
    print('stddev accuracy: {:.3f}, stddev f1-score: {:.3f}'.format(vacc, vf1))