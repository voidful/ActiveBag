from classifier import *
from selector import *
from utilities import *

storage_params = {
    'training_file': "./test/training_data/taipei/training.data",
    'validation_file': "./test/training_data/taipei/testing.data",
    'testing_file': "./test/training_data/taipei/testing.data",
    'output_dir': "./output/"
}
storage = Storage(storage_params)

classifier_params = {
    'epoch': [5, 300],
    'word_ngrams': [1, 12],
    'loss': ["ns", "hs", "softmax"],
    'lr': [0.01, 0.99],
    'lr_update_rate': [5, 500],
    'dim': [100, 700],
    'bucket': [5000, 20000]
}
filter_params = {
    'f1': [0.5, 1]
}
classifier = NoUseClassifier(storage, classifier_params, filter_params)
classifier.init(5)
classifier.valid()
valid_file = classifier.predict_validation()
selector = MaxEntropySelector(valid_file)
print(selector.select(5))
