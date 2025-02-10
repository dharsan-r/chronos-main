#!/bin/bash

for i in {2..9}
do
    nohup python train.py --config configs/test/chronos-t5-small-$i.yaml &
    wait $!  # Wait for the background job to finish
done
