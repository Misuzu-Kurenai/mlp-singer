#! /bin/bash

input_dir=data/bin/train
output_dir=mel_inferred_dataset

for ptfile in $input_dir/*.pt; do
    base=$(basename $ptfile .pt)
    output_file=$output_dir/$base.npy
    python inference_koko.py --input $ptfile --output $output_file
done