#!/bin/bash
mapfile -t files < <(ls -v *slurm.batch_*)

for file in "${files[@]}"; do
    
    echo "$file"
    sbatch "$file"
    
done
