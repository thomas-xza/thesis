#!/usr/bin/env python3

from pathlib import Path
import lzma
import io

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor


import matplotlib.pyplot as plt


torch.set_default_dtype(torch.bfloat16)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def run_utilisation_loop_batch(model: nn.Module, batch_name: str):

    print("loop back")

    directory = Path('./weights/')

    # Find all files with batch_name in filename
    
    for file in directory.glob(f"model_weights*{batch_name}*xz"):

        print(str(file))
        
        if file.is_file():
            
            dist_metadata = str(file).split(batch_name)[-1].replace('.xz','')

            graph_filename = f"{batch_name}{dist_metadata}"

            weights_path = f"./{str(file)}"

            run_utilisation_loop_once(model, weights_path, graph_filename)
            
            # Do something with the file

    pass


def run_utilisation_loop_once(model: nn.Module, weights_path: str, graph_filename: str):

    ##  For loop here to iterate over all weights available.

    ##  Decompress to memory.

    with lzma.open(weights_path, 'rb') as f:

        decompressed_data = f.read()

    ##  Load weights.

    checkpoint = torch.load(io.BytesIO(decompressed_data),
                           weights_only = True,
                           map_location=torch.device('cpu')
                           )

    old_state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    print(old_state_dict)
    
    ##  Fix naming scheme and load module.

    model.load_state_dict(
                {k.replace('module.', '', 1): v for k, v in old_state_dict.items()}
    )

    res = []
    
    interval = 1 / (2 ** 6)

    dl = DataLoader(
                TensorDataset(
                    torch.arange(1.57, 4.71 + interval, interval)
                ),
                batch_size = 1,
                shuffle = False
    )
    
    with torch.no_grad():

        # all_tensors = []

        # for i in range(1):

            outputs_set = []
            
            for batch, (X,) in enumerate(dl):

                X = X.to(device)

                # print(X)

                ##  Run X through model, generate prediction.

                y_hat = model(X)

                print(X.item(), y_hat.item())

                outputs_set.append((X.item(), y_hat.squeeze(0).item()))
                
            # inner_tensor = torch.stack(outputs_set, dim=0)

            # all_tensors.append(inner_tensor)

            # print(all_tensors)

            x, y = zip(*outputs_set)

            plt.figure(figsize=(6, 4))

            plt.plot(
                        x, y,
                        marker='o',
                        markersize=4,          # small dots
                        markerfacecolor='black',
                        markeredgecolor='black',
                        linewidth=0.8,         # thin connecting line
                        color='black',
                        label='Trajectory'
            )

            # plt.scatter(
            #             x, y,
            #             color='black',          # point colour
            #             edgecolor='black',          # optional border around each marker
            #             s=10,                       # marker size
            #             label='Samples'
            # )
            
            plt.xlabel('Input value')
            plt.ylabel('Probability of π classification')
            plt.grid(True, ls='--', lw=0.5, alpha=0.7)
            plt.tight_layout()

            plt.tight_layout()

            plt.savefig(f"{graph_filename}.pdf", dpi=300)
            plt.close()      

    
