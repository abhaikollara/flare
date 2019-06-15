import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from flare import Trainer

class SingleInputModel(nn.Module):

    def __init__(self):
        super(SingleInputModel, self).__init__()
        self.l1 = nn.Linear(5, 2)

    def forward(self, x):
        return self.l1(x)

model = SingleInputModel()
train_x = np.random.randn(1000, 5).astype("float32")
train_y = np.random.randint(0, 2, size=(1000, ))

trainer = Trainer(model, F.nll_loss, Adam(model.parameters()))

class TestTrainer(object):

    def test_train(self):
        trainer.train(train_x, train_y)

    def test_evaluate(self):
        trainer.evaluate(train_x, train_y)

    def test_predict(self):
        trainer.predict(train_x)