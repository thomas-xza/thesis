#!/usr/bin/env python3
import shutil
import os

import random

original_file = "py_ex_4_gpuL.slurm.epochs_01_kaiming_02"

for i in range(10):

    rseed = random.randint(1, 1000000)

    epochs = str(50)

    new_filename = original_file.replace("01", epochs).replace("02", str(rseed))

    shutil.copy(original_file, new_filename)

    with open(original_file, 'r') as f:

        content = f.read()

    edited_content = content.replace("EPOCHS", epochs).replace("RSEED", str(rseed))

    with open(new_filename, 'w') as f:

        f.write(edited_content)

