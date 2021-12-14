
## Start

### Available Environment
```bash
MountainCar-v0 DeepSea-v0, Acrobot-v1
```

### Default Setting

```bash
--sample-per-step False --same-noise-update True --max-step 500 --epoch 1000 --num-actoms 51 --v-max 100
## for DeepSea-v0
--size 10
```
### HyperC51 with prior 

```bash
## Greedy action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 2.0 --noise-dim 2 --ensemble-num 0 --action-sample-num 0 
## MAX action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 2.0 --noise-dim 2 --ensemble-num 0 --action-sample-num 20 --action-select-scheme "MAX"
## VIDS action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 2.0 --noise-dim 2 --ensemble-num 0 --action-sample-num 20 --action-select-scheme "VIDS"
## DeepSea-v0 with VIDS
$ python -m new_rainbow --task DeepSea-v0 --seed 2021 --v-max 10 --prior-std 2.0 --noise-dim 2 --ensemble-num 0 --action-sample-num 20 --action-select-scheme "VIDS"
```

### NoisyC51 with prior 

```bash
## Greedy action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 1.0 --noise-dim 0 --ensemble-num 0 --action-sample-num 0 
## MAX action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 1.0 --noise-dim 0 --ensemble-num 0 --action-sample-num 20 --action-select-scheme "MAX"
## VIDS action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 1.0 --noise-dim 0 --ensemble-num 0 --action-sample-num 20 --action-select-scheme "VIDS"
## DeepSea-v0 with VIDS
$ python -m new_rainbow --task DeepSea-v0 --seed 2021 --v-max 10 --prior-std 1.0 --noise-dim 0 --ensemble-num 0 --action-sample-num 20 --action-select-scheme "VIDS"
```

### EnsembleC51 with prior 

```bash
## Greedy action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 1.0 --noise-dim 0 --ensemble-num 20 --action-sample-num 0 
## MAX action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 1.0 --noise-dim 0 --ensemble-num 20 --action-sample-num 20 --action-select-scheme "MAX"
## VIDS action select
$ python -m new_rainbow --task MountainCar-v0 --seed 2021 --prior-std 1.0 --noise-dim 0 --ensemble-num 20 --action-sample-num 20 --action-select-scheme "VIDS"
## DeepSea-v0 with VIDS
$ python -m new_rainbow --task DeepSea-v0 --seed 2021 --v-max 10 --prior-std 1.0 --noise-dim 0 --ensemble-num 20 --action-sample-num 20 --action-select-scheme "VIDS"
```


### *DNQ with prior 
```
$ python -n new_rainbow --num-atoms 1
```