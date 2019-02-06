from math import log

from base import baseSelector


# selected_and_store_retrain_data

class MaxEntropySelector(baseSelector.BaseSelector):
    prob_denominator = 0

    def group_values_in_class(self, values):
        self.prob_denominator += sum(values)
        return sum(values)

    def group_classes_to_result(self, classes_value):
        total_entropy = 0
        for val in classes_value:
            prob = val / self.prob_denominator
            total_entropy += prob * log(prob, 2)
        return -total_entropy
