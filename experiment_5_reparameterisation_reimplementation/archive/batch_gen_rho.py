#!/usr/bin/env python3

import shutil
import os

import random

original_file = "py_ex_5_gpuL_test.slurm.batch_01"


with open(original_file, 'r') as f:
                
    content = f.read()

    for j in range(0, 1):

        print(-3 + j * 0.2)

        m_rho = str(-3 + j * 0.05)

        mu_rho_set = content.replace("MU_RHO_SET", m_rho)

        for k in range(0, 1):

            print(-3 + k * 0.2)

            b_rho = str(-3 + k * 0.5)

            mu_b_rho_set = mu_rho_set.replace("B_RHO_SET", b_rho)

            batch_n = f"mrho_{j}_brho_{k}"

            print(batch_n)
            
            wname_mu_b_rho_set = mu_b_rho_set.replace("WNAME_SET", batch_n)

            new_filename = original_file.replace("01", batch_n)

            shutil.copy(original_file, new_filename)

            with open(new_filename, 'w') as f:

                f.write(wname_mu_b_rho_set)
