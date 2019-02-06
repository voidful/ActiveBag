from collections import defaultdict


class BaseSelector:

    def group_values_in_class(self, values):
        pass

    def group_classes_to_result(self, classes_value):
        pass

    def __init__(self, valid_result):
        self.valid_result = valid_result

    def select(self, number, decent=True):
        for inputs, pred_list in self.valid_result.items():
            c_dict = defaultdict(list)
            v_list = []
            for i in pred_list:
                c_dict[i['predict']].append(i['prob'])
            for values in c_dict.values():
                v_list.append(self.group_values_in_class(values))
            self.valid_result[inputs] = self.group_classes_to_result(v_list)
        return sorted(self.valid_result.items(), key=lambda kv: kv[1], reverse=decent)[:number]

    def vote(self, pick=1, decent=True):
        result = defaultdict(float)
        for inputs, pred_list in self.valid_result.items():
            c_dict = defaultdict(list)
            for i in pred_list:
                c_dict[i['predict']].append(i['prob'])
            for key, values in c_dict.items():
                result[key] = self.group_values_in_class(values)
        return sorted(result.items(), key=lambda kv: kv[1], reverse=decent)[:pick]
