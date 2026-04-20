#!/usr/bin/env python3

import argparse
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import Pi_dataset
from mini_model_deterministic import Linear_model
from optimiser_sgd_deterministic import SGD_det

parser = argparse.ArgumentParser()
parser.add_argument('--neurons',
                    required = True,
                    type = int,
                    help = 'Quantity of n neurons in 1-n-n-1 network.')
args = parser.parse_args()

##  Initialisation.

torch.set_default_dtype(torch.bfloat16)

torch.manual_seed(239852)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def main():
    ##  Load up data.

    batch_size_pi = 2

    train_data = Pi_dataset("pi_dataset_10000.txt")

    train_dataloader = DataLoader(train_data, batch_size = batch_size_pi,
                                  shuffle=True)

    test_data = Pi_dataset("pi_dataset_2000.txt")

    test_dataloader = DataLoader(test_data, batch_size = batch_size_pi,
                                 shuffle=True)

    ##  Define model.

    model = Linear_model(args.neurons).to(device)

    # print(model.state_dict())

    ##  Define loss.

    loss_0 = nn.BCELoss()

    ##  Define parameter optimisation mechanism.

    optimizer = SGD_det(model.parameters(), lr=1e-3)

    epochs = 5

    start_time = time.time()

    while time.time() - start_time < 300:
    # for t in range(epochs):

        # print(f"Epoch {t+1}\n-------------------------------")

        train(train_dataloader, model, loss_0, optimizer)

        test(test_dataloader, model, loss_0)

    print(f"{time.time() - start_time}s")

    # print(model.state_dict())

    

def train(train_dataloader: DataLoader, model: nn.Module, loss_0:
          nn.Module, optimizer: torch.optim.Optimizer):

    ##  Setup parameters for loss function.

    size = len(train_dataloader.dataset)
    
    model.train()
    
    for batch, (X, y) in enumerate(train_dataloader):
        
        X, y = X.to(device), y.to(device)

        ##  Compute prediction error.
        
        y_hat = model(X)

        loss_res = loss_0(y_hat, y)
        
        ##  Backpropagation.
        
        loss_res.backward()

        ##  Iterate parameters then reset graph.
        
        optimizer.step()
        
        optimizer.zero_grad()

        if batch % 10000 == 0:

          # print(model.state_dict().keys())

          # print(model.state_dict()['linear_relu_stack.0.weight'][0])
          # print(model.state_dict()['linear_relu_stack.0.weight'][0].grad)

          current = (batch + 1) * len(X)
            
          # print(torch.stack([X, y, y_hat], dim=0).T)

          # print(torch.round(y_hat), y)

          # print(loss_res.item())

          # print(f"[{current:>5d}/{size:>5d}]")

          # print(f"loss_1: {loss_1:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader, model: nn.Module, loss_fn:
         nn.Module):

    ##  Setup parameters for loss function.

    model.eval()

    test_loss, correct = 0, 0
    
    with torch.no_grad():
        
        for X, y in dataloader:
            
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    
    correct /= len(dataloader.dataset)
    
    print(f"test_loss: {test_loss:>8f}")
            

main()
