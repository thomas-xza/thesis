#!/bin/bash
mapfile -t files < <(ls -v *_kaiming_*)

for file in "${files[@]}"; do
    
    sbatch "$file"
    
done
