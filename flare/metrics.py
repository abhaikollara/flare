import numpy as np
import torch


class Metric(object):

    def call(self, outputs, targets):
        raise NotImplementedError

    def __call__(self, outputs, targets):
        return self.call(outputs, targets)
    
    def reset_states(self):
        self.value = 0.0
    
    def batch_update(self, logs):
        raise NotImplementedError

    def __str__(self):
        try:
            return self.name
        except AttributeError:
            self.name = type(self).__name__
            return self.name

class AverageMetric(Metric):

    def batch_update(self, logs):
        batch_no = logs['batch_no']
        y = logs['y']
        Y = logs['Y']
        batch_val = self.call(y, Y).item()
        self.value = (self.value * batch_no + batch_val)/ (batch_no + 1)

class Loss(Metric):

    def batch_update(self, logs):
        batch_val = logs['batch_loss']
        batch_no = logs['batch_no']
        self.value = (self.value * batch_no + batch_val)/ (batch_no + 1)


class Accuracy(AverageMetric):

    def call(self, outputs, targets):
        y = torch.argmax(outputs, -1)

        return torch.mean((y == targets).float())
