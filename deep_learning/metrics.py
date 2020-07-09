import numpy as np


def true_positive(prediction, target):
    return ((prediction > 0) * (target > 0)).float().sum()


def false_positive(prediction, target):
    return ((prediction > 0) * (target <= 0)).float().sum()


def false_negative(prediction, target):
    return ((prediction <= 0) * (target > 0)).float().sum()


def true_negative(prediction, target):
    return ((prediction <= 0) * (target <= 0)).float().sum()


AGGREGATORS = {
    'TP': true_positive,
    'FP': false_positive,
    'FN': false_negative,
    'TN': true_negative,
}


class Metrics():
    def __init__(self, *metrics):
        self.metrics = metrics
        self.required_aggregators = set()
        for m in self.metrics:
            self.required_aggregators |= m.required_aggregators()
        self.reset()

    def reset(self):
        self.state = {}
        self.running_agg = {}
        self.running_count = {}

    def step(self, prediction, target, **additional_terms):
        for agg in self.required_aggregators:
            if agg not in self.state:
                self.state[agg] = AGGREGATORS[agg](prediction, target)
            else:
                self.state[agg] += AGGREGATORS[agg](prediction, target)

        for term in additional_terms:
            if term not in self.running_agg:
                self.running_agg[term] = additional_terms[term]
                self.running_count[term] = 1
            else:
                self.running_agg[term] += additional_terms[term]
                self.running_count[term] += 1

    def evaluate(self):
        values = {}
        for m in self.metrics:
            values[m.__name__] = m.evaluate(self.state)
        for key in self.running_agg:
            values[key] = self.running_agg[key] / self.running_count[key]
        self.reset()
        return values


class Accuracy():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP', 'FN', 'TN'])

    @staticmethod
    def evaluate(state):
        correct = state['TP'] + state['TN']
        wrong = state['FP'] + state['FN']
        if correct + wrong == 0:
            return np.nan
        return correct / (correct + wrong)


class Precision():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP'])

    @staticmethod
    def evaluate(state):
        if state['TP'] + state['FP'] == 0:
            return np.nan
        return state['TP'] / (state['TP'] + state['FP'])


class Recall():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FN'])

    @staticmethod
    def evaluate(state):
        if state['TP'] + state['FN'] == 0:
            return np.nan
        return state['TP'] / (state['TP'] + state['FN'])


class F1():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP', 'FN', 'TN'])

    @staticmethod
    def evaluate(state):
        precision = Precision.evaluate(state)
        recall = Recall.evaluate(state)
        if precision + recall == 0:
            return np.nan
        return 2 * precision * recall / (precision + recall)


class IoU():
    @staticmethod
    def required_aggregators():
        return set(['TP', 'FP', 'FN'])

    @staticmethod
    def evaluate(state):
        intersection = state['TP']
        union = state['TP'] + state['FN'] + state['FP']
        return intersection / union
