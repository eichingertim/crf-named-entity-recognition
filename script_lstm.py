import json
import random
from tqdm import tqdm

from pre_processor import PreProcessor

from collections import Counter

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Input

import numpy as np

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

"""
PREPROCSESSING
"""

extraction_of = 'A'

vocabs, kmeans = PreProcessor.calculateClusters(data)

print("START PREPROCESSING")
data = PreProcessor.preprocess_tf(data, extraction_of)


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w,  t in zip(s['token'].values.tolist(), 
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

words = list(set(vocabs))
words.append("ENDPAD")

n_words = len(words)

tags = ['O', extraction_of]
n_tags = len(tags)

word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

getter = SentenceGetter(data)
sentences = getter.sentences

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=140, sequences=X, padding="post",value=n_words - 1)

y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=n_tags) for i in y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)


input = Input(shape=(140,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=140)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)

model = Model(input, out)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=3, validation_split=0.2, verbose=1)

y_true = []
y_pred = []

for i in tqdm(range(0, len(X_test))):
    p = model.predict(np.array(X_test[i]))
    p = np.argmax(p, axis=-1)

    t = y_test[i]

    for wordIdx, trueTagIdx, predTagIdx in zip(X_test[i], t, p[0]):
        token = words[wordIdx]
        tag_true = tags[np.argmax(trueTagIdx, axis=-1)]
        tag_pred = tags[predTagIdx]

        y_true.append(tag_true)
        y_pred.append(tag_pred)

        # print(f"W: {words[wordIdx]}, T: {tags[np.argmax(trueTagIdx, axis=-1)]}, P: {tags[predTagIdx]}")

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
