class PreProcessor:
    
    @staticmethod
    def preprocess_test(extraction_of, data):
        sentences = list()
        for i,(k,v) in enumerate(data.items()):
            tokens = v.get('tokens')
            labeler = [val for s, val in data[k].items() if s != 'tokens']

            for j in range(0, len(labeler)):
                sentence = dict()
                sentence['tokens'] = tokens
               
                labels_from_all = [l.get(extraction_of) for l in labeler]
                sentence['labeling'] = labels_from_all[j]
                sentences.append(sentences)

        return sentences

    @staticmethod
    def preprocess_training(extraction_of, data):
        sentences = list()
        for k,v in data.items():

            sentence = dict()
            sentence['tokens'] = v.get('tokens')

            labeler = [val for s, val in data[k].items() if s != 'tokens']

            if (len(labeler) == 0):
                continue


            labels_from_all = [l.get(extraction_of) for l in labeler]
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

            sentence['labeling'] = one_merged_label
            sentences.append(sentence)

        return sentences

