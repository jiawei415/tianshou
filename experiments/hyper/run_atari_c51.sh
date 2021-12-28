export CUDA_VISIBLE_DEVICES=$1
seed=$2

## environment config
task=$3
## training config
same_noise_update=True
batch_noise_update=True
target_update_freq=10000
batch_size=320
lr=0.0001
weight_decay=0
n_step=3
v_max=100
num_atoms=51
## algorithm config
alg_type=hyper
noise_std=1
noise_dim=32
noise_norm=0
target_noise_std=0
hyper_reg_coef=0.01
hyper_weight_decay=0
prior_std=1
prior_scale=0.1
posterior_scale=0.1
## action selection config
sample_per_step=False
action_sample_num=1
action_select_scheme=Greedy
value_gap_eps=0.001
value_var_eps=0.001
## network config
hidden_layer=1
hidden_size=512
use_dueling=True
use_dueling=1
is_double=1
init_type=''
## epoch config
epoch=100
step_per_epoch=50000
step_per_collect=4
## buffer config
buffer_size=1000000
min_buffer_size=${batch_size}

## overwrite config
## W. Prior -- Sample per epiosde -- Dependent noise update
config01="{'prior_std':${prior_std},'sample_per_step':False,'same_noise_update':True}"
## W. Prior -- Sample per step -- Dependent noise update
config02="{'prior_std':${prior_std},'sample_per_step':True,'same_noise_update':True}"
## W. Prior -- Sample per epiosde -- Independent noise update
config03="{'prior_std':${prior_std},'sample_per_step':False,'same_noise_update':False}"
## W. Prior -- Sample per step -- Independent noise update
config04="{'prior_std':${prior_std},'sample_per_step':True,'same_noise_update':False}"
## W/O. Prior -- Sample per epiosde -- Dependent noise update
config05="{'prior_std':0,'sample_per_step':False,'same_noise_update':True}"
## W/O. Prior -- Sample per step -- Dependent noise update
config06="{'prior_std':0,'sample_per_step':True,'same_noise_update':True}"
## W/O. Prior -- Sample per epiosde -- Independent noise update
config07="{'prior_std':0,'sample_per_step':False,'same_noise_update':False}"
## W/O. Prior -- Sample per step -- Independent noise update
config08="{'prior_std':0,'sample_per_step':True,'same_noise_update':False}"

config="${config01}"

if [ -z "$task" ]; then
  task="PongNoFrameskip-v4"
fi


time=2
for i in $(seq 5)
do
    tag=$(date "+%Y%m%d%H%M%S")
    python -m tianshou.scripts.run_${alg_type} --seed ${seed} --task ${task} \
    --target-update-freq=${target_update_freq} --batch-size=${batch_size} --lr=${lr} \
    --weight-decay=${weight_decay} --n-step=${n_step} --v-max=${v_max} --num-atoms=${num_atoms} \
    --noise-std=${noise_std} --noise-dim=${noise_dim} --noise-norm=${noise_norm} \
    --target-noise-std=${target_noise_std} --hyper-reg-coef=${hyper_reg_coef} --hyper-weight-decay=${hyper_weight_decay} \
    --prior-std=${prior_std} --prior-scale=${prior_scale} --posterior-scale=${posterior_scale} \
    --action-sample-num=${action_sample_num} --action-select-scheme=${action_select_scheme} \
    --value-gap-eps=${value_gap_eps} --value-var-eps=${value_var_eps} \
    --hidden-layer=${hidden_layer} --hidden-size=${hidden_size} \
    --use-dueling=${use_dueling} --is_double=${is_double} --init-type=${init_type} \
    --epoch=${epoch} --step-per-epoch=${step_per_epoch} --step-per-collect=${step_per_collect} \
    --buffer-size=${buffer_size} --min-buffer-size=${min_buffer_size} \
    --config ${config} \
    > ~/logs/${alg_type}_${task}_${tag}_3.out 2> ~/logs/${alg_type}_${task}_${tag}_3.err &
    echo "run $seed $tag"
    let seed=$seed+1
    sleep ${time}
done

