#!/bin/bash

cmd=$1
task=$2
seed=$3

v_max=0
num_atoms=1
## action selection config
sample_per_step=False
action_sample_num=1
action_select_scheme=Greedy
value_gap_eps=0.001
value_var_eps=0.001

task="${task}NoFrameskip-v4"
config="{
    'same_noise_update':True,'batch_noise_update':True,'target_update_freq':10000,'batch_size':320,'lr':0.0001,'weight_decay':0,'n_step':3,'v_max':${v_max},'num_atoms':${num_atoms}, \
    'noise_std':1,'noise_dim':32,'noise_norm':0,'target_noise_std':0,'hyper_reg_coef':0.01,'hyper_weight_decay':0,'prior_std':1,'prior_scale':0.1,'posterior_scale':0.1, \
    'sample_per_step':${sample_per_step},'action_sample_num':${action_sample_num},'action_select_scheme':'${action_select_scheme}','value_gap_eps':${value_gap_eps},'value_var_eps':${value_var_eps}, \
    'hidden_layer':1,'hidden_size':512,'use_dueling':True,'use_dueling':1,'is_double':1,'init_type':'', \
    'epoch':100,'step_per_epoch':50000,'step_per_collect':4, \
    'buffer_size':1000000,'min_buffer_size':50000
}"

if [ "$cmd" = 'kill' ]; then
    if [ -z "$task" ]; then
        echo "kill tianshou"
        ps -ef | grep tianshou | awk '{print $2}'| xargs kill -9
    fi
    if [ -n "$task" ]; then
        echo "kill $task"
        ps -ef | grep $task | awk '{print $2}'| xargs kill -9
    fi
fi

if [ "$cmd" = 'run' ]; then
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/root/miniconda3/etc/profile.d/conda.sh"
        else
            export PATH="/root/miniconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    conda activate torch

    python /data/jiawei/Code/tianshou/tianshou/scripts/run_hyper_atari.py --task $task --seed $seed \
        --config "$config" --logdir '/data/jiawei/Code/tianshou/results' > log.out 2> log.err &

fi
