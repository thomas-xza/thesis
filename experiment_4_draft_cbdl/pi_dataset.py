#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset, DataLoader


class Pi_dataset(Dataset):

    
    def __init__(self, filename):

        with open(filename, 'r') as f:

            line_count = sum(1 for line in f)

            # print(line_count)
            
            self.data = torch.zeros((line_count, 1), dtype=torch.bfloat16)

            self.labels = torch.zeros((line_count, 1), dtype=torch.bfloat16)

            
        with open(filename, 'r') as f:

            for i, line in enumerate(f):

                a = line.strip().split(',')

                self.data[i] = float(a[0])

                self.labels[i] = float(a[1])

                
    def __len__(self):

        return len(self.data)

    
    def __getitem__(self, idx):

        return self.data[idx], self.labels[idx]
