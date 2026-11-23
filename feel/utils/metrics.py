import statistics

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from collections import defaultdict


class Metric:
    """
    Use the values class to calculate various metrics
    """
    def __init__(self, metrics):
        # record the output values and target values of the model for each batch
        self.values = {}
        self.outputs = []
        self.targets = []
        self.losses = []
        self.metrics = metrics

    def accuracy(self):
        self.values['acc'] = accuracy_score(self.targets,self.outputs)
        # calculate the accuracy
        return self.values['acc']

    def update(self, outputs, targets, loss=None):
        # append one batch outputs and targets to all outputs and targets
        if torch.is_tensor(outputs):
            self.outputs += outputs.cpu().detach().tolist()
            self.targets += targets.cpu().detach().tolist()
        else:
            self.outputs += outputs.tolist()
            self.targets += targets.tolist()
        if loss is not None:
            self.losses.append(loss)

    def macro_f1_score(self):
        self.values['macro-f1'] = f1_score(self.targets, self.outputs, average='macro')
        # calculate the macro f1-score
        return self.values['macro-f1']

    def micro_f1_score(self):
        self.values['micro-f1'] = f1_score(self.targets, self.outputs, average='micro')
        # calculate the micro f1-score
        return self.values['micro-f1']

    def weighted_f1_score(self):
        self.values['weighted-f1'] = f1_score(self.targets, self.outputs, average='weighted')
        return self.values['weighted-f1']

    def ck_coe(self):
        self.values['ck'] = cohen_kappa_score(self.targets, self.outputs)
        # calculate the micro f1-score
        return self.values['ck']
    def value(self):
        if not self.targets:
            return "No data to evaluate."
        # 　if one hot code, then transform to ordinary label
        if type(self.targets[0]) is list:
            try:
                self.targets = [self.targets[i].index(1) for i in range(len(self.targets))]
            except ValueError:
                return "unavailable"
        func = {
            'acc': self.accuracy,
            'macro-f1': self.macro_f1_score,
            'micro-f1': self.micro_f1_score,
            'ck': self.ck_coe,
            'weighted-f1': self.weighted_f1_score,
        }
        out = ""
        for m in self.metrics:
            out += f"{m}: {func[m]():.3f}   "
        if len(self.losses) != 0:
            return out + f"loss: {sum(self.losses)/len(self.losses):.4f}"
        else:
            return out

class SubMetric(Metric):
    def __init__(self,metrics):
        super().__init__(metrics=metrics)
        self.sub_labels = []
    def update(self, outputs, targets, sub_labels, loss=None):
        # append one batch outputs and targets to all outputs and targets
        if torch.is_tensor(outputs):
            self.outputs += outputs.cpu().detach().tolist()
            self.targets += targets.cpu().detach().tolist()
            self.sub_labels += sub_labels.cpu().detach().tolist()
        else:
            self.outputs += outputs.tolist()
            self.targets += targets.tolist()
            self.sub_labels += sub_labels.cpu().detach().tolist()
        if loss is not None:
            self.losses.append(loss)
    def sub_accuracy(self):
        # this function will return every subject's acc
        groups = defaultdict(lambda : {"outputs":[], "targets":[]})
        for o, t, s in zip(self.outputs, self.targets, self.sub_labels):
            groups[s]["outputs"].append(o)
            groups[s]["targets"].append(t)
        result = {}
        for label in groups:
            acc = accuracy_score(groups[label]["targets"], groups[label]["outputs"])
            result[label] = acc
        return result
    def sub_macro_f1_score(self):
        # this function will return every subject's macro-f1
        groups = defaultdict(lambda: {"outputs": [], "targets": []})
        for o, t, s in zip(self.outputs, self.targets, self.sub_labels):
            groups[s]["outputs"].append(o)
            groups[s]["targets"].append(t)
        result = {}
        for label in groups:
            acc = f1_score(groups[label]["targets"], groups[label]["outputs"], average='macro')
            result[label] = acc
        return result

    def accuracy(self):
        """
        this function will return the mean and std acc of all subjects
        :return:
        """
        acc_dict =  list(self.sub_accuracy().values())
        self.values['acc'] = statistics.mean(acc_dict)
        if len(acc_dict) != 1:
            self.values['acc_std'] = statistics.stdev(acc_dict)
        return self.values['acc']

    def macro_f1_score(self):
        """
        this function will return the mean and std macro-f1 of all subjects
        :return:
        """
        macro_f1_s = list(self.sub_macro_f1_score().values())
        self.values['macro-f1'] = np.mean(macro_f1_s)
        if len(macro_f1_s) != 1:
            self.values['macro-f1_std'] =  np.std(macro_f1_s)
        return self.values['macro-f1']
