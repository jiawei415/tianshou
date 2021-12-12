#!/bin/bash

cmd=$1
task=$2
config=$3

# echo "cmd: $cmd, task: $task"

if [ "$cmd" = 'kill' ]; then
    if [ -z "$task" ]; then
        echo "kill tianshou"
        ps -ef|grep tianshou |awk '{print $2}'|xargs kill -9
    fi
    if [ -n "$task" ]; then
        echo "kill $task"
        ps -ef|grep $task |awk '{print $2}'|xargs kill -9
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
    
    python /data/jiawei/Code/tianshou/new_rainbow.py --task $task --config $config > log.out 2> log.err &
    # python /data/jiawei/Code/tianshou/other.py --task $task --config $config > log.out 2> log.err &
    # echo "run $task"
fi
