import numpy as np
import torch


class Metric(object):

    def call(self, outputs, targets):
        raise NotImplementedError

    def __call__(self, outputs, targets):
        return self.call(outputs, targets)


class Accuracy(Metric):

    def call(self, outputs, targets):
        y = torch.argmax(outputs, -1)

        return torch.mean((y == targets).float())