import json
import random

from pre_processor import PreProcessor

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

split_parameter = round(len(keys)*0.7)
keys_train = keys[:split_parameter]
keys_test = keys[split_parameter: ]

train_data = dict()
for k in keys_train:
    train_data[k] = data[k]

test_data = dict()
for k in keys_test:
    test_data[k] = data[k]

"""
EXTRACTION DECLARATION
"""

extraction_of = 'sentiments'
# extraction_of = 'modifiers'
# extraction_of = 'aspects'

"""
PREPROCSESSING
"""

train_data = PreProcessor.preprocess_training(extraction_of, train_data)
test_data = PreProcessor.preprocess_test(extraction_of, test_data)

train_data_len = len(train_data)
print(f"LENGTH TEST-DATA: ${train_data_len}")

test_data_len = len(test_data)
print(f"LENGTH TEST-DATA: ${test_data_len}")