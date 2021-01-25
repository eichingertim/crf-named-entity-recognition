import json
import random

from sklearn_crfsuite import metrics
from collections import Counter

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
                        v2_new = [l.replace('B_','').replace('I_','').replace('I','').replace('B','') for l in v2]
                        v1_new[k2] = v2_new
                else:
                    v_new[k1] = v1
    return data_new

data = load_data('data_laptop_absa.json')

sentiment_crf = CRFModel('S', data)
aspects_crf = CRFModel('A', data)
modifiers_crf = CRFModel('M', data)

s_pred, s_test = sentiment_crf.fit_and_predict()
a_pred, a_test = aspects_crf.fit_and_predict()
m_pred, m_test = modifiers_crf.fit_and_predict()


f1_sentiments = metrics.flat_f1_score(s_pred, s_test)
f1_aspects = metrics.flat_f1_score(a_pred, a_test)
f1_modifiers = metrics.flat_f1_score(m_pred, m_test)


print(f"F1-Score Sentiments {f1_sentiments}")
print(f"F1-Score Aspects {f1_aspects}")
print(f"F1-Score Modfiiers {f1_modifiers}")

print(f"F1-Score Average {(f1_sentiments + f1_aspects + f1_modifiers)/3}")