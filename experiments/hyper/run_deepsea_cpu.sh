#!/bin/bash

cmd=$1
task=$2
config=$3

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

    python /data/jiawei/Code/tianshou/tianshou/scripts/run_hyper.py --task $task --seed 2021 --size 20 \
    --target-update-freq=4 --batch-size=128 --lr=0.001 --weight-decay=0 --n-step=1 --v-max=0 --num-atoms=1 --num-quantiles=1 \
    --noise-std=1 --noise-dim=2 --noise-norm=1 --target-noise-std=0 --hyper-reg-coef=0 --hyper-weight-decay=0 \
    --prior-std=1 --prior-scale=10 --posterior-scale=1 \
    --action-sample-num=1 --action-select-scheme='Greedy' --value-gap-eps=0.001 --value-var-eps=0.001 \
    --hidden-layer=2 --hidden-size=64 --use-dueling=0 --is_double=1 --init-type='trunc_normal' \
    --epoch=1000 --step-per-collect=1 --buffer-size=200000 --min-buffer-size=20 \
    --config "$config" --logdir '/data/jiawei/Code/tianshou/results' > log.out 2> log.err &

fi
