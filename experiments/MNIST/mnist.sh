#!/bin/bash

epochs=1
experiment_name=MNIST
threads=2
init_nets=8
max_nets=16
init_species=8
max_species=8

log_dir=./logs

if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi

python3 mnist.py --epochs $epochs \
                 --experiment-name $experiment_name \
                 --max-threads $threads \
                 --init-nets $init_nets \
                 --max-nets $max_nets \
                 --init-species $init_species \
                 --max-species $max_species \
                 --log-dir $log_dir
