#!/usr/bin/env python3

import argparse
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from pi_dataset import Pi_dataset
from mini_model import Linear_model

from experiment_training_lib import run_training_loop
from experiment_utilise_lib import run_utilisation_loop_once, run_utilisation_loop_batch


##  Initialisation.

parser = argparse.ArgumentParser()

parser.add_argument('--neurons-n',
                    required = True,
                    type = int,
                    help = 'Quantity of n neurons in 1-n-n-1 network. 512, or 1024 recommended.')

parser.add_argument('--batch-n',
                    required = True,
                    type = int,
                    help = 'Size of batch from dataset during training/testing. 256 recommended.')

parser.add_argument('--training-epochs',
                    required = True,
                    type = int,
                    help = 'Quantity of training epochs to run.')

parser.add_argument('--weights-name',
                    required = False,
                    type = str,
                    help = 'Name to append to stored weights filename.')

parser.add_argument('--weights-load',
                    required = False,
                    type = str,
                    help = 'Filename of stored weights to load.')

parser.add_argument('--weights-load-batch',
                    required = False,
                    type = str,
                    help = 'Portion of filename to match, to iterate over a set of different model weights.')

parser.add_argument('--uniform-init',
                    required = False,
                    default = 0.5,
                    type = float,
                    help = 'Initial value to set all model parameters to, as tuple.')

parser.add_argument('--random-seed',
                    required = False,
                    default = 239852,
                    type = int,
                    help = 'Random seed to use, such as when initialising weights, shuffling data.')

parser.add_argument('--w-rho-init',
                    required = False,
                    default = -3.0,
                    type = float,
                    help = 'Rho weight initialisation.')

parser.add_argument('--b-rho-init',
                    required = False,
                    default = -3.0,,
                    type = float,
                    help = 'Rho bias initialisation.')

args = parser.parse_args()

for key, value in vars(args).items():
            print(f"{key}: {value}")

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(args.random_seed)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def main():

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ##  Setup filename to store weights to.

    if args.random_seed != 239852:

        filename = f"model_weights_batch_{args.batch_n}_epochs_{args.training_epochs}_kaiming_{args.random_seed}"

    else:

        filename = f"model_weights_batch_{args.batch_n}_epochs_{args.training_epochs}_udist_{args.uniform_init}"


    ##  Define model.

    model = Linear_model(
                args.neurons_n,
                (args.uniform_init, args.uniform_init),
                args.random_seed,
                args.w_rho_init,
                args.b_rho_init
    ).to(device)

    ##  Ensure multiple GPUs used if available.
    
    if torch.cuda.device_count() > 1:
        
        print(f"Using {torch.cuda.device_count()} GPUs!")
        
        model = nn.DataParallel(model, device_ids=[0, 1])

    ##  Either load weights or load training data.

    if args.weights_load is not None:
        
        run_utilisation_loop(model, args.weights_load)

    elif args.weights_load_batch is not None:

        run_utilisation_loop_batch(model, args.weights_load_batch)

    else:

        ##  Load up limited training data to induce epistemic uncertainty.

        train_dl = DataLoader(
                    Subset(
                        Pi_dataset("pi_dataset_40000.txt"),
                        range(40000)
                    ),
                    batch_size = args.batch_n,
                    shuffle=True
        )

        test_dl = DataLoader(
                    Subset(
                        Pi_dataset("pi_dataset_8000.txt"),
                        range(8000)
                    ),
                    batch_size = args.batch_n,
                    shuffle=True
        )

        run_training_loop(model, train_dl, test_dl, args.training_epochs, filename)


if __name__ == '__main__':
            main()

