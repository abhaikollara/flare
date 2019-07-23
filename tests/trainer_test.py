import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from flare import Trainer
from flare.metrics import Accuracy
import pytest


class SingleInputModel(nn.Module):

    def __init__(self):
        super(SingleInputModel, self).__init__()
        self.l1 = nn.Linear(5, 3)
        self.l2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.l2(self.l1(x))

model = SingleInputModel()
train_x = np.random.randn(10000, 5).astype("float32")
train_y = (train_x.sum(axis=1) > 0).astype('int')
test_x = np.random.randn(100, 5).astype("float32")
test_y = (test_x.sum(axis=1) > 0).astype('int')


class TestTrainer(object):

    def test_train(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])
        history = trainer.train(train_x, train_y, epochs=2)
        train_logs = history.train_logs
        assert train_logs['Loss'][0] > train_logs['Loss'][1]

    def test_train_with_validation_data(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])
        history = trainer.train(train_x, train_y, epochs=2, validation_data=(test_x, test_y))
        train_logs, test_logs = history.train_logs, history.test_logs

        assert train_logs['Loss'][0] > train_logs['Loss'][1]
        assert test_logs['Loss'][0] > test_logs['Loss'][1]
        

    def test_train_with_validation_split(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])
        history = trainer.train(train_x, train_y, epochs=2, validation_split=0.2)
        train_logs, test_logs = history.train_logs, history.test_logs

        assert train_logs['Loss'][0] > train_logs['Loss'][1]
        assert test_logs['Loss'][0] > test_logs['Loss'][1]

    def test_train_generator(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])

    def test_evaluate(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])
        trainer.evaluate(train_x, train_y)

    def test_evaluate_generator(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])

    def test_predict(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])
        trainer.predict(train_x)
    
    def test_predict_generator(self):
        model = SingleInputModel()
        trainer = Trainer(model, nn.CrossEntropyLoss(), Adam(model.parameters()), metrics=[Accuracy()])