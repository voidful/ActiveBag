from nlp2 import *
from base import baseClassifier


class NoUseClassifier(baseClassifier.BaseClassifier):

    def train(self, **args):
        with open(args['out_file'], "w") as f:
            f.write(" ")

    def test(self, file_loc):
        result = {
            "precision": uniform(0, 1),
            "recall": uniform(0, 1),
            "f1": uniform(0, 1)
        }
        return result

    def predict_one(self, input_str, classifier_loc):
        return {
            "predict": randint(0, 10),
            "prob": uniform(0, 1)
        }
