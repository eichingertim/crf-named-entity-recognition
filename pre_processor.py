import pandas as pd
import spacy
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn import cluster

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nlp = spacy.load("en_core_web_sm")

class PreProcessor:

    LABELINGS = [
        'aspects',
        'sentiments',
        'modifiers'
    ]

    @staticmethod
    def stop_words(sentences):
        stopwordsList = stopwords.words('english')
        more_stopwords = ['.', ';', '!', ',', '*', '?', '/', '-', '"', '..']
        stopwordsList.extend(more_stopwords)


        for sen in sentences:
            for i, t in enumerate(sen['tokens']):
                if t in stopwordsList and not t in ['very', 'biggest', 'big', 'highly', 'high', 'not', 'cannot']:
                    sen['labeling'][i] = 'O'

        return sentences

    @staticmethod
    def lemmatize_sen(sen):
        l = WordNetLemmatizer()

        return [l.lemmatize(t) for t in sen]

    @staticmethod
    def calculateClusters(data):
        print("STARTED CALCULATING CLUSTERS")
        sentences = [PreProcessor.lemmatize_sen(v.get('tokens')) for k, v in data.items()]

        word2vec_model = Word2Vec(sentences, min_count=1)

        X = word2vec_model[word2vec_model.wv.vocab]

        num_clusters = 30

        kmeans = cluster.KMeans(n_clusters=num_clusters)
        kmeans.fit(X)

        print("FINISHED CALCULATING CLUSTERS")
        return word2vec_model.wv.vocab, kmeans

    @staticmethod
    def preprocess_test_crf(data, getCluster, extraction_of):
        sentences = list()
        for k,v in tqdm(data.items()):
            tokens = PreProcessor.lemmatize_sen(v.get('tokens'))
            detokenzied = ' '.join(tokens)
            pos_tagged = nlp(detokenzied)
            pos = [t.tag_ for t in pos_tagged]
            labeler = [val for s, val in data[k].items() if s != 'tokens']

            for j in range(0, len(labeler)):
                sentence = dict()
                sentence['sentenceId'] = k+'|'+ str(j)
                sentence['tokens'] = tokens
                sentence['pos'] = pos

                eo = ''
                if extraction_of == 'S':
                    eo = 'sentiments'
                elif extraction_of == 'A':
                    eo = 'aspects'
                else:
                    eo = 'modifiers'

                sentence['labeling'] = labeler[j].get(eo)
                sentences.append(sentence)
        
        sentences = PreProcessor.stop_words(sentences)

        words = list()
        for s in sentences:
            for i, t in enumerate(s['tokens']):
                word = dict()
                word['id'] = s['sentenceId']
                word['token'] = t
                word['cluster'] = getCluster(t)
                word['pos'] = s['pos'][i]
                word['labeling'] = s['labeling'][i]
                if s['labeling'][i] == 'O' or s['labeling'][i] == extraction_of:
                    words.append(word)
                else:
                    word['labeling'] = 'O'
                    words.append(word)
                words.append(word)

        return pd.DataFrame(words)

    @staticmethod
    def preprocess_train_crf(data, getCluster, extraction_of):
        sentences = list()
        for k,v in tqdm(data.items()):

            labeler = [val for s, val in data[k].items() if s != 'tokens']

            if (len(labeler) == 0):
                continue

            merged_labelings = dict()
            for label in PreProcessor.LABELINGS:

                labels_from_all = [l.get(label) for l in labeler]

                one_merged_label = list()
                for i in range(len(labels_from_all[0])):
                    counter = dict()
                    for j in range(len(labels_from_all)):
                        try:
                            counter[labels_from_all[j][i]] += 1
                        except:
                            counter[labels_from_all[j][i]] = 1

                    most_labeled = ("O", 0)
                    for ke, va in counter.items():
                        if (most_labeled[1] < va):
                            most_labeled = (ke, va)

                    one_merged_label.append(most_labeled[0])   

                merged_labelings[label] = one_merged_label                             

            sentence = dict()
            sentence['sentenceId'] = k
            sentence['tokens'] = PreProcessor.lemmatize_sen(v.get('tokens'))
            detokenzied = ' '.join(sentence['tokens'])
            pos_tagged = nlp(detokenzied)
            sentence['pos'] = [t.tag_ for t in pos_tagged]
            sentence['labeling'] = [PreProcessor.getLabel(merged_labelings, i) for i in range(len(sentence['tokens']))]
            sentences.append(sentence)

        sentences = PreProcessor.stop_words(sentences)


        words = list()
        for s in sentences:
            for i, t in enumerate(s['tokens']):
                word = dict()
                word['id'] = s['sentenceId']
                word['token'] = t
                word['cluster'] = getCluster(t)
                word['pos'] = s['pos'][i]
                word['labeling'] = s['labeling'][i]
                if s['labeling'][i] == 'O' or s['labeling'][i] == extraction_of:
                    words.append(word)
                else:
                    word['labeling'] = 'O'
                    words.append(word)
                words.append(word)

        return pd.DataFrame(words)

    @staticmethod
    def getLabel(labeling, i):
        if labeling['sentiments'][i] == 'S':
            return 'S'
        elif labeling['aspects'][i] == 'A':
            return 'A'
        elif labeling['modifiers'][i] == 'M':
            return 'M'
        else:
            return 'O'
