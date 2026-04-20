#!/bin/bash
11;rgb:f6f6/f8f8/fafa
mapfile -t files < <(ls -v *_batch_*)

for file in "${files[@]}"; do
    
    echo "$file"
    
done
