#!/usr/bin/env python3

import shutil
import os

from decimal_range import decimal_range

original_file = "py_ex_4_gpuL.slurm.epochs_01_udist_02"

##  Uniform distribution loop.

for i in list(decimal_range('0.0625', '0.5', '0.0625')):

    udist = str(i)

    ##  Training epochs loop.

    epochs = "50"
    
    new_filename = original_file.replace("01", str(epochs)).replace("02", udist)

    shutil.copy(original_file, new_filename)

    with open(new_filename, 'r') as f:

        content = f.read()

    edited_content = content.replace("EPOCHS", str(epochs)).replace("UDIST", udist)

    with open(new_filename, 'w') as f:

        f.write(edited_content)

