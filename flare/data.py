import torch
from torch.utils.data import Dataset

def convert_to_tensor(inp):
    if isinstance(inp, torch.Tensor):
        return inp
    elif isinstance(inp, (list, tuple)):
        return [torch.from_numpy(x) for x in inp]
    else:
        return torch.from_numpy(inp)

def to_list(inp):
    if isinstance(inp, (list, tuple)):
        return inp
    else:
        return [inp]

#
# Is there a cleaner way to do this ?
#
class FlareDataset(Dataset):

    def __init__(self, inputs, targets=None):
        self.inputs = convert_to_tensor(inputs)
        if targets is not None:
            self.targets = convert_to_tensor(targets)
        else:
            self.targets = None

        # For models with multiple inputs and targets
        if len({l for l in to_list(self.inputs)}) > 1:
            raise ValueError("n_samples dimensions must be sample for all inputs")

        if self.targets is not None:
            if len({l for l in to_list(self.targets)}) > 1:
                raise ValueError("n_samples dimensions must be sample for all inputs")

            # Check n_samples of inputs and targets
            if len(to_list(self.inputs)[0]) != len(to_list(self.targets)[0]):
                raise ValueError("n_samples dimensions of inputs and targets must be same")


    def __getitem__(self, idx):
        if isinstance(self.inputs, list):
            x = [inputs[idx] for inputs in self.inputs]
        else:
            x = self.inputs[idx]

        if self.targets is not None:
            if isinstance(self.targets, list):
                y = [targets[idx] for targets in self.targets]
            else:
                y = self.targets[idx]

                return (x, y)
        
        return x
    
    def __len__(self):
        return len(to_list(self.inputs)[0])