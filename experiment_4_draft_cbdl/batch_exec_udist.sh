#!/bin/bash
mapfile -t files < <(ls -v *_udist_*)

for file in "${files[@]}"; do
    
    sbatch "$file"
    
done
