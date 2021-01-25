import json
import random

from pre_processor import PreProcessor

import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter

random.seed(1234321)

def load_data(filename):
    with open(filename,'r', encoding='utf8') as infile:
        data = json.load(infile)
        data_new = dict()
        for k,v in list(data.items()):
            v_new = data_new.setdefault(k,dict())
            for k1,v1 in list(v.items()):
                v1_new = v_new.setdefault(k1,dict())
                if k1 != 'tokens':
                    for k2,v2 in list(v1.items()):
                        v2_new = [l.replace('B_','').replace('I_','').replace('I','').replace('B','') for l in v2]
                        v1_new[k2] = v2_new
                else:
                    v_new[k1] = v1
    return data_new

data = load_data('data_laptop_absa.json')

keys = list(data.keys())
random.shuffle(keys)

split_parameter = round(len(keys)*0.8)
keys_train = keys[:split_parameter]
keys_test = keys[split_parameter: ]

train_data = dict()
for k in keys_train:
    train_data[k] = data[k]

test_data = dict()
for k in keys_test:
    test_data[k] = data[k]

"""
PREPROCSESSING
"""

vocabs, kmeans = PreProcessor.calculateClusters(data)

def getCluster(token):
    words = list(vocabs)
    return kmeans.labels_[words.index(token)]

extraction_of = 'A'

print("START TRAINING PREPROCESSING")
train_data = PreProcessor.preprocess_train_crf(train_data, getCluster, extraction_of)

print("START TEST PREPROCESSING")
test_data = PreProcessor.preprocess_test_crf(test_data, getCluster, extraction_of)

train_data_len = len(train_data)
print(train_data[:5])
print(f"LENGTH TRAIN-DATA: {train_data_len}")

test_data_len = len(test_data)
print(test_data[:5])
print(f"LENGTH TEST-DATA: {test_data_len}")

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

getter_train = SentenceGetter(train_data)
sentences_train = getter_train.sentences

getter_test = SentenceGetter(test_data)
sentences_test = getter_test.sentences

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

X_train = [sent2features(s) for s in sentences_train]

y_train = [sent2labels(s) for s in sentences_train]

X_test= [sent2features(s) for s in sentences_test]
y_test = [sent2labels(s) for s in sentences_test]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.0001,
    c2=0.0001,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)

print(metrics.flat_classification_report(y_test, y_pred))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))
print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(30))
print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-30:])