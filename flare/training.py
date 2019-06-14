import torch


class Trainer(object):

    def __init__(self, model, loss, optimizer, device="cpu"):
        self.model = model
        self.loss_fn = loss
        self.optimizer = optimizer
        self.device = torch.device(device)
    
    def train(self):
        pass
    
    def evaluate(self):
        pass
    
    def predict(self):
        pass

    def train_generator(self):
        pass
    
    def evaluate_generator(self):
        pass
    
    def predict_generator(self):
        pass

    def train_batch(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()

        return loss
    
    def evaluate_batch(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        output = self.model(inputs)
        loss = self.loss_fn(output, targets)

        return loss
    
    def predict_batch(self, inputs):
        inputs = inputs.to(self.device)

        output = self.model(inputs)

        return output

