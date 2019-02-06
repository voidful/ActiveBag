from classifier import *
from selector import *
from utilities import *

storage_params = {
    'id': '1549449878F7ED',
    'retrain_file': "./test/training_data/taipei/retrain.data"
}
storage = Storage(storage_params)

classifier = NoUseClassifier(storage)
pred_result = classifier.predict("testing")
print(pred_result)

selector = MaxEntropySelector(pred_result)
print(selector.vote())
