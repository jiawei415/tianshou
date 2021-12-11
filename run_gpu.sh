#!/bin/bash

task=$1
seed=$3

epoch=300
size=10
max_step=500
num_atoms=51
v_max=100
prior_std=2.0
noise_dim=2
ensemble_num=0
sample_per_step=False
same_noise_update=True
action_sample_num=0
action_select_scheme='MAX'
batch_noise_update=True

config="{
    'size':${size}, \
    'max_step':${max_step}, \
    'epoch':${epoch}, \
    'num_atoms':${num_atoms}, \
    'v_max':${v_max}, \
    'prior_std':${prior_std}, \
    'noise_dim':${noise_dim}, \
    'ensemble_num':${ensemble_num}, \
    'sample_per_step':${sample_per_step}, \
    'same_noise_update':${same_noise_update}, \
    'action_sample_num':${action_sample_num}, \
    'action_select_scheme':'${action_select_scheme}', \
    'batch_noise_update':${batch_noise_update}
}"

time=2
export CUDA_VISIBLE_DEVICES=$2

for i in $(seq 5)
do
    tag=$(date "+%Y%m%d%H%M%S")
    python hyper_rainbow.py --task ${task} --seed ${seed} --config "${config}" > ~/logs/${task}_${tag}_3.out 2> ~/logs/${task}_${tag}_3.err &
    echo "run $seed $tag"
    let seed=$seed+1
    sleep ${time}
done

# ps -ef | grep ${task} | awk '{print $2}'| xargs kill -9
# ps -ef | grep tianshou | awk '{print $2}'| xargs kill -9

# MountainCar-v0, Acrobot-v1, DeepSea-v0
