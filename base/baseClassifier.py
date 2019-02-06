import threading
from collections import defaultdict

from nlp2 import *


class BaseClassifier:

    def train(self, **args):
        pass

    def load_classifier(self, loc):
        pass

    def test(self):
        pass

    def predict_one(self):
        pass

    def __init__(self, storage, training_params=None, filter_params=None):
        self.storage = storage
        self.training_params = training_params
        self.filter_params = filter_params
        self.classifier_params = defaultdict(dict)

    def init(self, num_of_classifier=0, in_iter=False):
        train_detail = self._cal_parameter(num_of_classifier, self.training_params)
        self._train_classifiers_with_multi_thread(train_detail)
        filter_result = self.filter(train_detail, self.filter_params)
        if len(filter_result) < num_of_classifier:
            re_filter_result = self.init(num_of_classifier - len(filter_result), in_iter=True)
            filter_result.update(re_filter_result)
        if not in_iter:
            for i in self.storage.list_all_classifier():
                if not is_list_contain_string(i, [*filter_result]):
                    self.storage.remove_classifier(i)
            self.classifier_params = filter_result
            self.storage.write_classifier_param(filter_result)
        return filter_result

    def retrain(self):
        self.storage.tidy_retrain_data()
        train_detail = self.storage.load_classifier_param()
        self._train_classifiers_with_multi_thread(train_detail)
        self.classifier_params = train_detail
        return train_detail

    def filter(self, classifier_obj, filter_parameter):
        detail_list = self.valid(classifier_obj)
        keep_list = defaultdict(dict)
        for key, value in filter_parameter.items():
            for cn, cd in detail_list.items():
                if value[0] < cd[key] < value[1]:
                    keep_list[cn] = cd
        return dict(keep_list)

    def valid(self, classifier_params=None):
        if classifier_params is None:
            classifier_params = self.classifier_params
        _classifiers = defaultdict(dict)
        for name, detail in classifier_params.items():
            detail.update(self.test(detail['out_file']))
            _classifiers[name] = detail
        return dict(_classifiers)

    def predict(self, input_pred):
        result_dict = defaultdict(list)
        for name, detail in self.classifier_params.items():
            one_sample_result = self.predict_one(input_pred, self.load_classifier(detail['out_file']))
            one_sample_result['classifier'] = name
            result_dict[input_pred].append(one_sample_result)
        return dict(result_dict)

    def predict_validation(self):
        result_dict = defaultdict(list)
        valid = read_files_into_lines(self.storage.get_validation_file())
        for i in valid:
            result_dict.update(self.predict(i))
        return dict(result_dict)

    def _cal_parameter(self, model_num, parameter):
        train_detail = defaultdict(dict)
        for i in range(0, model_num):
            param = dict()
            for key, value in parameter.items():
                param[key] = random_value_in_array_form(value)
            param['train_file'] = self.storage.get_training_file()
            filename = random_string(7)
            param['out_file'] = self.storage.get_classifier_dir() + filename
            train_detail[filename] = param
        return dict(train_detail)

    def _train_classifiers_with_multi_thread(self, parameter):
        thread_list = []
        for key, value in parameter.items():
            p = threading.Thread(target=self.train,
                                 kwargs=value)
            thread_list.append(p)
        for p in thread_list:
            p.start()
            p.join()

    # def predict_bagging(self, classifiers, input_sentence):
    #     mixed_output_result = defaultdict(int)
    #     classifier_output_prob = defaultdict(list)
    #     prob_denominator = 0
    #     for detail in self.predict_multi(classifiers, input_sentence):
    #         prob_denominator += detail.max_label_prob
    #         classifier_output_prob[detail.max_label] = detail.max_label_prob
    #     for key, value in classifier_output_prob.items():
    #         mixed_output_result[key] = sum(value) / prob_denominator
    #
    #     result = defaultdict(str)
    #     result.max_label_prob = mixed_output_result[max(mixed_output_result, key=mixed_output_result.get)]
    #     result.max_label = max(mixed_output_result, key=mixed_output_result.get)
    #     result.entropy = stats.entropy(list(mixed_output_result.values()))
    #     return result
