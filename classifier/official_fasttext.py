import fasttext

from base import baseClassifier


class OfficialFastTextClassifier(baseClassifier.BaseClassifier):

    def load_classifier(self, file_loc):
        if '.bin' not in file_loc:
            file_loc += '.bin'
        return fasttext.load_model(file_loc, label_prefix='__label__')

    # download pretrain wordvector form here
    # https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
    # input_file             training file path (required)
    # output                 output file path (required)
    # label_prefix           label prefix ['__label__']
    # lr                     learning rate [0.1]
    # lr_update_rate         change the rate of updates for the learning rate [100]
    # dim                    size of word vectors [100]
    # ws                     size of the context window [5]
    # epoch                  number of epochs [5]
    # min_count              minimal number of word occurences [1]
    # neg                    number of negatives sampled [5]
    # word_ngrams            max length of word ngram [1]
    # loss                   loss function {ns, hs, softmax} [softmax]
    # bucket                 number of buckets [0]
    # minn                   min length of char ngram [0]
    # maxn                   max length of char ngram [0]
    # thread                 number of threads [12]
    # t                      sampling threshold [0.0001]
    # silent                 disable the log output from the C++ extension [1]
    # encoding               specify input_file encoding [utf-8]
    # pretrained_vectors     pretrained word vectors (.vec file) for supervised learning []
    def train(self, train_file, out_file, epoch=300, word_ngrams=7, loss='hs', lr=0.01, lr_update_rate=10, dim=300,
              bucket=200000):
        fasttext.supervised(train_file, out_file, label_prefix='__label__', silent=False, lr=lr,
                            lr_update_rate=lr_update_rate, epoch=epoch, dim=dim, bucket=bucket, word_ngrams=word_ngrams,
                            loss=loss)

    def test(self, file_loc):
        test_result = self.load_classifier(file_loc).test(self.storage.get_testing_file())
        result = {
            "precision": test_result.precision,
            "recall": test_result.recall,
            "f1": (2 * test_result.precision * test_result.recall) / (test_result.precision + test_result.recall)
        }
        return result

    def predict_one(self, input_str, classifier):
        p = classifier.predict_proba([input_str])[0][0]
        return {
            "class": p[0],
            "prob": p[1]
        }
