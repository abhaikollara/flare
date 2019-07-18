import torch
from torch.utils.data import DataLoader
from .data import FlareDataset
from .callbacks import CallbackList, Baselogger, MetricLogger
from .metrics import Accuracy, Metric
import math
import time


class Trainer(object):

    def __init__(self, model, loss, optimizer, device="cpu", metrics=None):
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.device = torch.device(device)

        # Callbacks and metrics
        self.metrics = metrics
        self.history = Baselogger(metrics=self.metrics)
        self.callbacks = CallbackList(self, [self.history])
        
        if self.metrics is not None:
            if not isinstance(self.metrics[0], Metric):# Check for all
                raise TypeError("metric must be an instance of Metric")
            else:
                self.callbacks.append(MetricLogger(self.metrics))
    
    def train(self, inputs, targets, epochs=1, batch_size=32, shuffle=True):
        dataset = FlareDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return self.train_generator(dataloader, epochs=epochs)
    
    def evaluate(self, inputs, targets, batch_size=32):
        dataset = FlareDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return self.evaluate_generator(dataloader)
    
    def predict(self, inputs, batch_size=32):
        dataset = FlareDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return self.predict_generator(dataloader)

    def train_generator(self, dataloader, epochs=1):
        self.callbacks.on_train_begin()

        n_batches = len(dataloader.dataset)/dataloader.batch_size
        if dataloader.drop_last:
            n_batches = int(n_batches)
        else:
            n_batches = math.ceil(n_batches)

        logs = {'n_batches':n_batches}
        for i in range(epochs):
            self.callbacks.on_epoch_begin(logs={'epoch':i})

            for batch_no, (x, Y) in enumerate(dataloader):
                logs['batch_no'] = batch_no
                self.callbacks.on_train_batch_begin(logs=logs)
                batch_loss, y = self.train_batch(x, Y)
                logs['batch_loss'] = batch_loss
                logs['y'] = y
                logs['Y'] = Y
                self.callbacks.on_train_batch_end(logs=logs)

            self.callbacks.on_epoch_end(logs={'epoch':i})
        
        self.callbacks.on_train_end()
    
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
            logs['batch_loss'] = batch_loss
            logs['y'] = y
            logs['Y'] = Y
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

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            output = self.model(inputs)
            loss = self.loss_fn(output, targets)

        return loss.item(), output
    
    def predict_batch(self, inputs):
        if self.model.training:
            self.model.eval()

        inputs = inputs.to(self.device)

        with torch.no_grad():
            output = self.model(inputs)

        return output

