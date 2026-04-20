#!/usr/bin/env python3

import argparse
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import Pi_dataset

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(239852)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def run_training_loop(model : nn.Module, train_dl: DataLoader, test_dl: DataLoader, epochs: int, filename: str):

    ##  Save model params to plaintext.

    with open("model_params_weight_mu_pre.txt", "w") as f:

        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        f.write(str(state_dict['linear_relu_stack.0.weight_mu']))

    ##  Define loss.

    loss = nn.BCELoss()

    ##  Define parameter optimisation mechanism.

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # while time.time() - start_time < 300:

    ##  Train either for fixed period of time or fixed quantity of epochs.

    start_time = time.time()
    ##  Output training start time.

    print(datetime.now().strftime("%H:%M:%S"))

    ##  Run train-test sequence.

    for t in range(epochs):

        print(f"Training...")

        train(train_dl, model, loss, optimizer)

        print(f"Testing...")

        test(test_dl, model, loss)

    ##  Output training time.

    print(f"{time.time() - start_time}s")

    # print(model.state_dict())

    ##  Save model params to plaintext.

    with open("model_params_weight_mu_pre.post", "w") as f:

        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        f.write(str(state_dict['linear_relu_stack.0.weight_mu']))

    with open("model_params_weight_mu_pre.post", "w") as f:

        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        f.write(str(state_dict['linear_relu_stack.0.weight_rho']))

    ##  Generate timestamp and store weights for later loading back.

    torch.save(model.state_dict(), filename)



def train(dl: DataLoader, model: nn.Module, loss: nn.Module,
          optimizer: torch.optim.Optimizer):

    ##  Put model into training mode; ensures gradient tracking.

    model.train()
    
    for batch, (X, y) in enumerate(dl):
        
        X, y = X.to(device), y.to(device)

        ##  Run X through model, generate prediction.
        
        y_hat = model(X)

        ##  Calculate error of prediction.

        # loss_res = loss(y_hat, y)

        # print(y_hat)

        try:
            loss_res = loss(y_hat, y)

        except:
            print("ERROR:", y_hat, y)
        
        ##  compute gradients numerically via backpropagation, back to
        ##  leaf nodes of graph.
        
        loss_res.backward()

        ##  Iterate parameters.
        
        optimizer.step()

        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        
        print(str(state_dict['linear_relu_stack.0.weight_rho']))
        
        ##  Reset gradients within graph.
        
        optimizer.zero_grad()

        # optimizer.debug_off()

        if batch % 1000 == 0:

            pass
                  
            # print(torch.stack([X, y, y_hat], dim=0).T)

            ##  It is not necessary to know the exact parameter
            ##  values, just that they are changing.

            # optimizer.debug_off()

            # print(model.state_dict().keys())

            # print(model.state_dict()['linear_relu_stack.0.weight_mu'].sum(),
            #       model.state_dict()['linear_relu_stack.0.rho_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.0.mu_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.0.rho_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.2.weight_mu'].sum(),
            #       model.state_dict()['linear_relu_stack.2.rho_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.2.mu_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.2.rho_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.4.weight_mu'].sum(),
            #       model.state_dict()['linear_relu_stack.4.rho_weight'].sum(),
            #       model.state_dict()['linear_relu_stack.4.mu_bias'].sum(),
            #       model.state_dict()['linear_relu_stack.4.rho_bias'].sum(),
            #       )

            # print(model.state_dict()['linear_relu_stack.0.weight_mu'][0])
            # print(model.state_dict()['linear_relu_stack.0.weight_mu'][0].grad)
            
            # print(torch.round(y_hat), y)

            # print(loss_res.item())
            
            # current = (batch + 1) * len(X)
            
            # print(f"[{current:>5d}/{len(train_dl.dataset):>5d}]")

            # print(f"loss_1: {loss_1:>7f}  [{current:>5d}/{size:>5d}]")


def test(dl: DataLoader, model: nn.Module, loss: nn.Module):

    ##  Setup parameters for loss function.

    model.eval()

    test_loss, correct = 0, 0
    
    with torch.no_grad():
        
        for X, y in dl:
            
            X, y = X.to(device), y.to(device)
            
            y_hat = model(X)

            test_loss += loss(y_hat, y).item()
            
            # correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
                    
    test_loss /= len(dl)
    
    # correct /= len(dl.dataset)
    
    print(f"{test_loss:>8f}")


