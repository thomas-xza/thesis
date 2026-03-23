#!/usr/bin/env python3

import argparse
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

import reparam


env = {
            "neurons_n": 1024,
            "batch_n": 64,
            "training_epochs": 1,
            "weights_name": "test",
            "random_seed": 239852,
            "mu_w_init": 0.1,
            "mu_b_init": -3.0,
            "rho_w_init": 0.1,
            "rho_b_init": -3.0,
}


for k, v in env.items():
            print(f"{k}: {v}")

torch.set_default_dtype(torch.float32)
torch.manual_seed(env["random_seed"])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ##  Setup filename to store weights to.

    filename = f"model_weights_batch_{env['batch_n']}_epochs_{env['training_epochs']}_{env['weights_name']}"


    ##  Define model.

    model = Linear_model(
                env["neurons_n"],
                env["mu_w_init"],
                env["mu_b_init"],
                env["rho_w_init"],
                env["rho_b_init"],
    ).to(device)

    ##  Ensure multiple GPUs used if available.
     
    if torch.cuda.device_count() > 1:
        
        print(f"Using {torch.cuda.device_count()} GPUs!")
        
        model = nn.DataParallel(model, device_ids=[0, 1])

    ##  Either load weights or load training data.

    if env["weights_load"] is not None:
        
        run_utilisation_loop_one(
                    model,
                    env["weights_load"],
                    env["sample_quantity"]
        )

    elif env["weights_load_batch"] is not None:

        run_utilisation_loop_batch(
                    model,
                    env["weights_load_batch"],
                    env["sample_quantity"]
        )

    else:

        ##  Load up limited training data to induce epistemic uncertainty.

        train_dl = DataLoader(
                    Subset(
                        Pi_dataset("pi_dataset_40000.txt"),
                        range(40000)
                    ),
                    batch_size = env["batch_n"],
                    shuffle=True
        )

        test_dl = DataLoader(
                    Subset(
                        Pi_dataset("pi_dataset_8000.txt"),
                        range(8000)
                    ),
                    batch_size = env["batch_n"],
                    shuffle=True
        )

        run_training_loop(model,
                          train_dl,
                          test_dl,
                          env["training_epochs"],
                          filename)


if __name__ == '__main__':
            main()

