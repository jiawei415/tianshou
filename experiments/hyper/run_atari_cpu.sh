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

    if [ -z "$task" ]; then
        task="Pong"
    fi
    task="${task}NoFrameskip-v4"

    if [ -z "$config" ]; then
        config="{}"
    fi

    python /data/jiawei/Code/tianshou/tianshou/scripts/run_hyper_atari.py --task $task --seed 2022 \
    --target-update-freq=10000 --batch-size=320 --lr=0.0001 --weight-decay=0 --n-step=3 --v-max=0 --num-atoms=1 --num-quantiles=1 \
    --noise-std=1 --noise-dim=32 --noise-norm=0 --target-noise-std=0.01 --hyper-reg-coef=0.01 --hyper-weight-decay=0 \
    --prior-std=1 --prior-scale=0.1 --posterior-scale=0.1 \
    --action-sample-num=1 --action-select-scheme='Greedy' --value-gap-eps=0.001 --value-var-eps=0.0000001 \
    --hidden-layer=1 --hidden-size=512 --use-dueling=0 --is_double=1 --init-type='' \
    --epoch=100 -step-per-epoch=50000 --step-per-collect=4 --buffer-size=1000000 --min-buffer-size=50000 \
    --config "$config" --logdir '/data/jiawei/Code/tianshou/results' > log.out 2> log.err &

fi
