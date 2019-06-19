import torch
from torch.utils.data import DataLoader
from .data import FlareDataset
import math
import time


class Trainer(object):

    def __init__(self, model, loss, optimizer, device="cpu"):
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.device = torch.device(device)
    
    def train(self, inputs, targets, epochs=1, batch_size=32, shuffle=True):
        dataset = FlareDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return self.train_generator(dataloader)
    
    def evaluate(self, inputs, targets, batch_size=32):
        dataset = FlareDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return self.evaluate_generator(dataloader)
    
    def predict(self, inputs, batch_size=32):
        dataset = FlareDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return self.predict_generator(dataloader)

    def train_generator(self, dataloader, epochs=1):
        avg_loss = 0.

        if dataloader.drop_last:
            n_batches = int(n_batches)
        else:
            n_batches = math.ceil(n_batches)

        for i in range(epochs):
            print("Epoch", i+1)
            for batch_no, (x, y) in enumerate(dataloader):
                batch_loss = self.train_batch(x, y)
                avg_loss = ((avg_loss * batch_no) + batch_loss)/ (batch_no + 1)
                #TODO: Seperate out print logic. Implement callbacks
                if batch_no > 0:
                    print("\r", end="")
                print(f" Batch {batch_no+1}/{n_batches} Loss {round(avg_loss,4)}", end="")
            print("\n")
        
        return avg_loss
    
    def evaluate_generator(self, dataloader):
        avg_loss = 0.
        print("Evaluating")
        n_batches = math.ceil(len(dataloader.dataset)/dataloader.batch_size)
        for batch_no, (x, y) in enumerate(dataloader):
            batch_loss = self.evaluate_batch(x, y)
            avg_loss += batch_loss
            #TODO: Seperate out print logic. Implement callbacks
        avg_loss /= n_batches
        print("Average Loss", round(avg_loss, 4))

        return avg_loss
    
    def predict_generator(self, dataloader):
        outs = []
        n_batches = math.ceil(len(dataloader.dataset)/dataloader.batch_size)
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
        return loss.item()
    
    def evaluate_batch(self, inputs, targets):
        if self.model.training:
            self.model.eval()

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        with torch.no_grad():
            output = self.model(inputs)
            loss = self.loss_fn(output, targets)

        return loss.item()
    
    def predict_batch(self, inputs):
        if self.model.training:
            self.model.eval()

        inputs = inputs.to(self.device)

        with torch.no_grad():
            output = self.model(inputs)

        return output

