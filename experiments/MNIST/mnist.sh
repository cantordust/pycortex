#!/bin/bash

epochs=10
experiment_name=MNIST
threads=2
init_nets=4
max_nets=16
init_species=2
max_species=4

log_dir=./logs

if [ ! -d "$log_dir" ]; then
	mkdir $log_dir
fi

python mnist.py --epochs $epochs \
                --experiment-name $experiment_name \
                --max-threads $threads \
                --init-nets $init_nets \
                --max-nets $max_nets \
                --init-species $init_species \
                --max-species $max_species \
                --log-dir $log_dir
