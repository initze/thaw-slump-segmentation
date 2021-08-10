import torch
import numpy as np
from typing import List


def nan_to_zero(ary):
    ary[ary.isnan()] = 0
    return ary


class Metrics():
    n_classes: int
    class_names: List[str]
    running_confusion_matrix: torch.Tensor

    def __init__(self, n_classes=None, class_names=None):
        self.class_names = class_names
        self.n_classes = n_classes

        if self.n_classes is None:
            self.n_classes = len(class_names)
        if self.class_names is None:
            self.class_names = [f'Class{i}' for i in range(self.n_classes)]

        self.reset()

    def reset(self):
        self.running_confusion_matrix = torch.zeros(self.n_classes, self.n_classes, dtype=torch.long)
        self.running_agg = {}
        self.running_count = {}

    def step(self, prediction, target, **additional_terms):
        # Make sure the CM is on the same device as our predictions
        self.running_confusion_matrix = self.running_confusion_matrix.to(prediction.device)

        prediction = prediction.argmax(dim=1)
        assert target.max() < self.n_classes, f"Number of target classes is larger than specified, please change the config accordingly! ({target.max()} >= {self.n_classes})"
        assert prediction.max() < self.n_classes, f"Number of predicted classes is larger than specified, this shouldn't happen! ({prediction.max()} >= {self.n_classes})"
        confusion_idx = target.flatten() + self.n_classes * prediction.flatten()
        batch_confusion_matrix = torch.bincount(confusion_idx, minlength=self.n_classes*self.n_classes)
        assert batch_confusion_matrix.shape[0] == self.n_classes*self.n_classes, f"Pytorch is doing weird stuff here... this shouldn't happen! {batch_confusion_matrix.shape[0]} != {self.n_classes*self.n_classes}"
        batch_confusion_matrix = batch_confusion_matrix.reshape(self.n_classes, self.n_classes)
        self.running_confusion_matrix += batch_confusion_matrix

        for term in additional_terms:
            if term not in self.running_agg:
                self.running_agg[term] = additional_terms[term]
                self.running_count[term] = 1
            else:
                self.running_agg[term] += additional_terms[term]
                self.running_count[term] += 1


    def evaluate(self):
        CM = self.running_confusion_matrix

        # Find out which classes are actually used
        # i.e. those that appear in the labels
        class_used = CM.sum(dim=0) > 0
        CM = CM[class_used][:, class_used]
        class_names = [c for c, u in zip(self.class_names, class_used) if u]

        values = {}

        # Accuracy: #TrueClassifications / #AllClassifications
        values['Accuracy'] = CM.diag().sum() \
                           / CM.sum()

        # Calculate TP, FP, FN, TN for each class
        # (these are vectors of length n_classes)
        TP = CM.diag()  # diagonal: prediction == ground_truth
        FP = CM.sum(dim=1) - TP  # FP = #PixelsActuallyInClass - TP
        FN = CM.sum(dim=0) - TP  # FN = #PixelsPredictedAsClass - TP
        TN = CM.sum() - TP - FP - FN  # FN = #Pixels - TP - FP - FN

        IoU = TP / (FP + FN + TP)
        values['mIoU'] = IoU.mean()

        Precision = nan_to_zero(TP / (TP + FP))
        Recall    = nan_to_zero(TP / (TP + FN))
        F1        = nan_to_zero(2 * (Precision * Recall) / (Precision + Recall))

        support = CM.sum(dim=0)  # Frequency of each class

        # Kappa calculation following
        # https://en.wikipedia.org/wiki/Cohen%27s_kappa#Definition
        N = CM.sum()
        p_o = CM.diag().sum() / N
        p_e = (CM.sum(dim=1) * CM.sum(dim=0)).sum() / (N*N)
        values['Kappa'] = (p_o - p_e) / (1 - p_e)

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        # There are different implementations of F1/Precision/Recall score for multiclass:
        values['F1_macro'] = F1.mean()  # == sklearn's f1_score(average='macro')
        values['F1_weighted'] = (F1 * support).sum() / support.sum() # == sklearn's f1_score(average='weighted')

        values['Precision_macro'] = Precision.mean()
        values['Precision_weighted'] = (Precision * support).sum() / support.sum()
        values['Recall_macro'] = Recall.mean()
        values['Recall_weighted'] = (Recall * support).sum() / support.sum()

        for i, classname in enumerate(class_names):
            values[f'{classname}_F1'] = F1[i]
            values[f'{classname}_Precision'] = Precision[i]
            values[f'{classname}_Recall'] = Precision[i]
            values[f'{classname}_IoU'] = IoU[i]

        metrics = {}

        for key in self.running_agg:
            metrics[key] = float(self.running_agg[key] / self.running_count[key])

        for key in values:
            metrics[key] = values[key].item()

        self.reset()
        return metrics 
