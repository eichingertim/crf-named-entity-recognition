import json
import random

from sklearn_crfsuite import metrics
from sklearn import metrics as m
from collections import Counter
from pre_processor import PreProcessor

from crf_model import CRFModel

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
                        v2_new = [l.replace('B_','').replace('I_','') \
                            .replace('I','').replace('B','') for l in v2]
                        v1_new[k2] = v2_new
                else:
                    v_new[k1] = v1
    return data_new

def load_test_ids(filename):
    with open(filename, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
        ids = list()
        ids.extend(data['test_IDs'])
    
    return ids

data = load_data('data_laptop_absa.json')
testIds = load_test_ids('test_IDs_laptop_absa.json')

# Caclulating clusters for data
cluster = PreProcessor.calculateClusters(data)

# Creating, Fitting and Predicting CRF Models

sentiment_crf = CRFModel('S', data, cluster)
aspects_crf = CRFModel('A', data, cluster)
modifiers_crf = CRFModel('M', data, cluster)

s_pred, s_test = sentiment_crf.fit_and_predict(testIds)
a_pred, a_test = aspects_crf.fit_and_predict(testIds)
m_pred, m_test = modifiers_crf.fit_and_predict(testIds)

#==================== Calculating metrics ===============================

# average=macro => unweighted score

recall_sentiments = m.recall_score(s_pred, s_test, average='macro')
precision_sentiments = m.precision_score(s_pred, s_test, average='macro')
f1_sentiments = m.f1_score(s_pred, s_test, average='macro')

recall_aspects = m.recall_score(a_pred, a_test, average='macro')
precision_spects = m.precision_score(a_pred, a_test, average='macro')
f1_aspects = m.f1_score(a_pred, a_test, average='macro')

recall_modifiers = m.recall_score(m_pred, m_test, average='macro')
precision_modifiers = m.precision_score(m_pred, m_test, average='macro')
f1_modifiers = m.f1_score(m_pred, m_test, average='macro')

print(f"Precision Sentiments {precision_sentiments}")
print(f"Recall Sentiments {recall_sentiments}")
print(f"F1-Score Sentiments {f1_sentiments}")

print(f"Precision Aspects {precision_spects}")
print(f"Recall Aspects {recall_aspects}")
print(f"F1-Score Aspects {f1_aspects}")

print(f"Precision Modifiers {precision_modifiers}")
print(f"Recall Modifiers {recall_modifiers}")
print(f"F1-Score Modfiiers {f1_modifiers}")

print(f"F1-Score Average {(f1_sentiments + f1_aspects + f1_modifiers)/3}")

#===================== Writing Learned Features ============================

def write_state_features(title, state_features, file):
    file.write(f"{title}\n")
    for (attr, label), weight in state_features:
        file.write("%0.6f %-8s %s\n" % (weight, label, attr))

file_s = open("testresults/sentiments_features.txt", "w")
f_s = list()
f_s.extend(Counter(sentiment_crf.crf.state_features_).most_common(15))
write_state_features("Top Sentiment Features", f_s, file_s)

file_a = open("testresults/aspects_features.txt", "w")
f_a = list()
f_a.extend(Counter(aspects_crf.crf.state_features_).most_common(15))
write_state_features("Top Aspect Features", f_a, file_a)

file_m = open("testresults/modifiers_features.txt", "w")
f_m = list()
f_m.extend(Counter(modifiers_crf.crf.state_features_).most_common(15))
write_state_features("Top Modifier Features", f_m, file_m)