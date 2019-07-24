import torch
from torch.utils.data import DataLoader
from .data import FlareDataset, convert_to_tensor
from .callbacks import CallbackList, Baselogger, MetricLogger
from .metrics import Accuracy, Metric, Loss
from .utils import split
import math
import time


class Trainer(object):

    def __init__(self, model, loss, optimizer, device="cpu", metrics=[]):
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.device = torch.device(device)

        # Callbacks and metrics
        self.metrics = [Loss(), *metrics]
        self.history = MetricLogger(self.metrics)
        self.callbacks = CallbackList(self, [Baselogger(metrics=self.metrics), self.history])

        for metric in self.metrics:
            if not isinstance(metric, Metric):# Check for all
                raise TypeError(f"metric must be an instance of Metric, found {type(metric)}")
    
    def train(self, inputs, targets, epochs=1, batch_size=32, validation_data=None, validation_split=None, shuffle=True):
        train_dataset = FlareDataset(inputs, targets)
        # import ipdb; ipdb.set_trace()
        if validation_data is None and validation_split is not None:
            (train_inputs, test_inputs), (train_targets, test_targets) = split([inputs, targets], validation_split)
            train_dataset = FlareDataset(train_inputs, train_targets)
            validation_data = (test_inputs, test_targets)

        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        return self.train_generator(dataloader, epochs=epochs, validation_data=validation_data)
    
    def evaluate(self, inputs, targets, batch_size=32):
        dataset = FlareDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return self.evaluate_generator(dataloader)
    
    def predict(self, inputs, batch_size=32):
        dataset = FlareDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return self.predict_generator(dataloader)

    def train_generator(self, dataloader, epochs=1, validation_data=None):
        if validation_data is not None:
            test_inputs, test_targets = validation_data
            test_dataset = FlareDataset(test_inputs, test_targets)
            test_dataloader = DataLoader(test_dataset, batch_size=dataloader.batch_size)

        self.callbacks.on_train_begin()

        if dataloader.drop_last:
            n_batches = int(len(dataloader.dataset)/dataloader.batch_size)
        else:
            n_batches = math.ceil(len(dataloader.dataset)/dataloader.batch_size)

        logs = {'n_batches':n_batches}
        for i in range(epochs):
            self.callbacks.on_epoch_begin(logs={'epoch':i})

            for batch_no, (x, Y) in enumerate(dataloader):
                logs['batch_no'] = batch_no
                self.callbacks.on_train_batch_begin(logs=logs)
                batch_loss, y = self.train_batch(x, Y)
                logs.update({'batch_loss':batch_loss, 'y':y, 'Y':Y})
                self.callbacks.on_train_batch_end(logs=logs)

            self.callbacks.on_epoch_end(logs={'epoch':i})
            if validation_data is not None:
                self.evaluate_generator(test_dataloader)
        self.callbacks.on_train_end()

        return self.history
    
    def evaluate_generator(self, dataloader):
        self.callbacks.on_eval_begin()
        n_batches = len(dataloader.dataset)/dataloader.batch_size
        if dataloader.drop_last:
            n_batches = int(n_batches)
        else:
            n_batches = math.ceil(n_batches)
        logs = {'n_batches': n_batches}
        for batch_no, (x, Y) in enumerate(dataloader):
            logs['batch_no'] = batch_no
            self.callbacks.on_eval_batch_begin(logs=logs)
            batch_loss, y = self.evaluate_batch(x, Y)
            logs.update({'batch_loss':batch_loss, 'y':y, 'Y':Y})
            self.callbacks.on_eval_batch_end(logs=logs)

        self.callbacks.on_eval_end()
    
    def predict_generator(self, dataloader):
        outs = []
        n_batches = len(dataloader.dataset)/dataloader.batch_size
        if dataloader.drop_last:
            n_batches = int(n_batches)
        else:
            n_batches = math.ceil(n_batches)

        print("Predicting")
        for batch_no, x in enumerate(dataloader):
            out = self.predict_batch(x)
            outs.append(out)
            if batch_no > 0:
                print("\r", end="")
            print(f"Batch {batch_no+1}/{n_batches}", end="")
        return torch.cat(outs)

    def train_batch(self, inputs, targets):
        if not self.model.training:
            self.model.train()

        inputs = convert_to_tensor(inputs)
        targets = convert_to_tensor(targets)

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), output
    
    def evaluate_batch(self, inputs, targets):
        if self.model.training:
            self.model.eval()

        inputs = convert_to_tensor(inputs)
        targets = convert_to_tensor(targets)

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            output = self.model(inputs)
            loss = self.loss_fn(output, targets)

        return loss.item(), output
    
    def predict_batch(self, inputs):
        if self.model.training:
            self.model.eval()

        inputs = convert_to_tensor(inputs)
        inputs = inputs.to(self.device)

        with torch.no_grad():
            output = self.model(inputs)

        return output
