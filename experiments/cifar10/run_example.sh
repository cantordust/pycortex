#!/bin/bash

runs=1
epochs=100
experiment_name=cifar10
max_workers=4
init_nets=8
max_nets=16
init_species=2
max_species=4
log_interval=200

data_dir="./data"
log_dir="./logs"

if [ ! -d "$log_dir" ]; then
    mkdir $log_dir
fi

mpirun --map-by core --np 4 \
python3 main.py \
--runs $runs \
--epochs $epochs \
--experiment-name $experiment_name \
--max-workers $max_workers \
--init-nets $init_nets \
--max-nets $max_nets \
--init-species $init_species \
--max-species $max_species \
--data-dir $data_dir \
--log-dir $log_dir \
--log-interval $log_interval \
--download-data \
--no-speciation \
--train-portion 0.05
