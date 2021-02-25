import random
import csv

from pre_processor import PreProcessor

import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import scorers

random.seed(1234321)

def flatten_array(ar):

    flattened = list()

    for s in ar:
        for t in s:
            if t == '1':
                print('lol')
            flattened.append(t)

    return flattened


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    cluster = sent[i][2]
        
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'cluster': cluster,
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        cluster1 = sent[i-1][2]
        features.update({
            '-1:word1.lower()': word1.lower(), 
            '-1:postag1': postag1,
            '-1:postag1[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        cluster1 = sent[i+1][2]
        features.update({
            '+1:word1.lower()': word1.lower(), 
            '+1:postag1': postag1,
            '+1:postag1[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
        
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, postag, cluster, label in sent]
def sent2tokens(sent):
    return [token for token, postag, cluster, label in sent]

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, c, t) for w, p, c, t in zip(s['token'].values.tolist(), 
                                                            s['pos'].values.tolist(), 
                                                            s['cluster'].values.tolist(),
                                                            s['labeling'].values.tolist())]
        self.grouped = self.data.groupby('id').apply(agg_func)
        self.sentences = [s for s in self.grouped]
            
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None

class CRFModel:

    def __init__(self, extraction_of, data):
        self.extraction_of = extraction_of
        self.data = data

        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.vocabs, self.kmeans = PreProcessor.calculateClusters(self.data)

        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c2=0.01,
            verbose=1,
            max_iterations=1000,
            all_possible_transitions=True
        )

    def getCluster(self, token):
        words = list(self.vocabs)
        return self.kmeans.labels_[words.index(token)]

    def fit_and_predict(self, keys_test):

        train_data = dict()
        test_data = dict()

        for k, v in self.data.items():
            if k in keys_test:
                test_data[k] = v
            else:
                train_data[k] = v

        print(f"START TRAINING PREPROCESSING FOR '{self.extraction_of}'")
        train_data = PreProcessor.preprocess_train_crf(train_data, self.getCluster, self.extraction_of)

        print(f"START TRAINING PREPROCESSING FOR '{self.extraction_of}'")
        test_data = PreProcessor.preprocess_test_crf(test_data, self.getCluster, self.extraction_of)

        getter_train = SentenceGetter(train_data)
        sentences_train = getter_train.sentences

        getter_test = SentenceGetter(test_data)
        sentences_test = getter_test.sentences

        self.X_train = [sent2features(s) for s in sentences_train]
        self.y_train = [sent2labels(s) for s in sentences_train]
        self.X_test= [sent2features(s) for s in sentences_test]
        self.y_test = [sent2labels(s) for s in sentences_test]

        print(f"START FITTING MODEL FOR '{self.extraction_of}'")
        self.crf.fit(self.X_train, self.y_train)
        print(f"FINISHED FITTING MODEL FOR '{self.extraction_of}'")

        iterations = self.crf.training_log_.iterations
        file = open('trainresults/' + self.extraction_of + "-train.csv", "w")

        file.write('loss,feature_norm\n')
        for it in iterations:
            file.write(str(it['loss']) + ',' + str(it['feature_norm']) + '\n')
        file.close()

        y_pred = self.crf.predict(self.X_test)

        file = open("testresults/" + self.extraction_of + "-test.csv", "w")

        abs_h = dict()

        for i, s in enumerate(self.X_test):
            for j, token in enumerate(s):

                label = y_pred[i][j]
                try:
                    abs_h[token['word.lower()']+"|"+label] += 1
                except:
                    abs_h[token['word.lower()']+"|"+label] = 1

        for t,n in abs_h.items():
            token = t.split("|")[0]
            label = t.split("|")[1]
            file.write(token + "," + label + "," + str(n) + "\n")
                

        return flatten_array(y_pred), flatten_array(self.y_test)