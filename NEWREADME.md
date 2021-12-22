
## Start

### Available Environment
```bash
MountainCar-v0 DeepSea-v0, Acrobot-v1
```

### Quick Experiments
```bash
## run hyper
pip install -e .
cd experiments/hyper
sh run_deepsea_c51.sh 0 2020 # 0 for cuda devices 2020 for seed
```

### HyperC51 with prior 

```bash
## Greedy action select
$ python -m tianshou.scripts.hyper --task MountainCar-v0 --seed 2021 --prior-std 1 --noise-dim 2 --v-max 100 --num-atoms 51 --action-sample-num 1 --action-select-scheme "Greedy"
## MAX action select
$ python -m tianshou.scripts.hyper --task MountainCar-v0 --seed 2021 --prior-std 1 --noise-dim 2 --v-max 100 --num-atoms 51 --action-sample-num 20 --action-select-scheme "MAX"
## VIDS action select
$ python -m tianshou.scripts.hyper --task MountainCar-v0 --seed 2021 --prior-std 1 --noise-dim 2 --v-max 100 --num-atoms 51 --action-sample-num 20 --action-select-scheme "VIDS"
## DeepSea-v0 with VIDS
$ python -m tianshou.scripts.hyper --task DeepSea-v0 --seed 2021 --prior-std 1 --noise-dim 2 --v-max 10 --num-atoms 51 --action-sample-num 20 --action-select-scheme "VIDS"
```

### NoisyC51 with prior 

```bash
## Greedy action select
$ python -m tianshou.scripts.noisy --task MountainCar-v0 --seed 2021 --prior-std 1 --noisy-std 0.1 --v-max 100 --num-atoms 51 --action-sample-num 1 --action-select-scheme "Greedy"
## MAX action select
$ python -m tianshou.scripts.noisy --task MountainCar-v0 --seed 2021 --prior-std 1 --noisy-std 0.1 --v-max 100 --num-atoms 51 --action-sample-num 20 --action-select-scheme "MAX"
## VIDS action select
$ python -m tianshou.scripts.noisy --task MountainCar-v0 --seed 2021 --prior-std 1 --noisy-std 0.1 --v-max 100 --num-atoms 51 --action-sample-num 20 --action-select-scheme "VIDS"
## DeepSea-v0 with VIDS
$ python -m tianshou.scripts.noisy --task DeepSea-v0 --seed 2021 --prior-std 1 --noisy-std 0.1 --v-max 10 --num-atoms 51 --action-sample-num 20 --action-select-scheme "VIDS"
```

### EnsembleC51 with prior 

```bash
## Greedy action select
$ python -m tianshou.scripts.ensemble --task MountainCar-v0 --seed 2021 --prior-std 1 --ensemble-num 20 --v-max 100 --num-atoms 51 --action-sample-num 1 --action-select-scheme "Greedy"
## MAX action select
$ python -m tianshou.scripts.ensemble --task MountainCar-v0 --seed 2021 --prior-std 1 --ensemble-num 20 --v-max 100 --num-atoms 51 --action-sample-num 20 --action-select-scheme "MAX"
## VIDS action select
$ python -m tianshou.scripts.ensemble --task MountainCar-v0 --seed 2021 --prior-std 1 --ensemble-num 20 --v-max 100 --num-atoms 51 --action-sample-num 20 --action-select-scheme "VIDS"
## DeepSea-v0 with VIDS
$ python -m tianshou.scripts.ensemble --task DeepSea-v0 --seed 2021 --prior-std 1 --ensemble-num 20 --v-max 10 --num-atoms 51 --action-sample-num 20 --action-select-scheme "VIDS"
```


### HyperDNQ with prior 
```
$ python -m tianshou.scripts.hyper --num-atoms 1
```