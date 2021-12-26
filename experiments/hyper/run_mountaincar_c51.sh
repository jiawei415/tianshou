export CUDA_VISIBLE_DEVICES=$1
seed=$2

## environment config
task=MountainCar-v0
max_step=500
## training config
same_noise_update=True
batch_noise_update=True
target_update_freq=100
batch_size=128
lr=0.0001
weight_decay=0.0003125
v_max=100
num_atoms=51
## algorithm config
alg_type=hyper
noise_std=1
noise_dim=2
target_noise_std=0
hyper_reg_coef=0.01
hyper_weight_decay=0
prior_std=1
prior_scale=1
## action selection config
sample_per_step=False
action_sample_num=1
action_select_scheme=Greedy
value_gap_eps=0.001
value_var_eps=0.001
## network config
hidden_layer=2
hidden_size=512
use_dueling=True
init_type=""
## epoch config
epoch=1000
step_per_collect=2
## buffer config
buffer_size=200000
min_buffer_size=${max_step}

## overwrite config
## W. Prior -- Sample per epiosde -- Dependent noise update
config01="{'prior_std':1,'sample_per_step':False,'same_noise_update':True,'use_dueling':${use_dueling}}"
## W. Prior -- Sample per step -- Dependent noise update
config02="{'prior_std':1,'sample_per_step':True,'same_noise_update':True,'use_dueling':${use_dueling}}"
## W. Prior -- Sample per epiosde -- Independent noise update
config03="{'prior_std':1,'sample_per_step':False,'same_noise_update':False,'use_dueling':${use_dueling}}"
## W. Prior -- Sample per step -- Independent noise update
config04="{'prior_std':1,'sample_per_step':True,'same_noise_update':False,'use_dueling':${use_dueling}}"
## W/O. Prior -- Sample per epiosde -- Dependent noise update
config05="{'prior_std':0,'sample_per_step':False,'same_noise_update':True,'use_dueling':${use_dueling}}"
## W/O. Prior -- Sample per step -- Dependent noise update
config06="{'prior_std':0,'sample_per_step':True,'same_noise_update':True,'use_dueling':${use_dueling}}"
## W/O. Prior -- Sample per epiosde -- Independent noise update
config07="{'prior_std':0,'sample_per_step':False,'same_noise_update':False,'use_dueling':${use_dueling}}"
## W/O. Prior -- Sample per step -- Independent noise update
config08="{'prior_std':0,'sample_per_step':True,'same_noise_update':False,'use_dueling':${use_dueling}}"

config="${config01}"

time=2
for i in $(seq 5)
do
    tag=$(date "+%Y%m%d%H%M%S")
    python -m tianshou.scripts.run_${alg_type} --seed ${seed} --task ${task} --max-step ${max_step} \
    --target-update-freq=${target_update_freq} --batch-size=${batch_size} --lr=${lr} \
    --weight-decay=${weight_decay} --v-max=${v_max} --num-atoms=${num_atoms} \
    --noise-std=${noise_std} --noise-dim=${noise_dim} --target-noise-std=${target_noise_std} \
    --prior-std=${prior_std} --prior-scale=${prior_scale} \
    --hyper-reg-coef=${hyper_reg_coef} --hyper-weight-decay=${hyper_weight_decay} \
    --action-sample-num=${action_sample_num} --action-select-scheme=${action_select_scheme} \
    --value-gap-eps=${value_gap_eps} --value-var-eps=${value_var_eps} \
    --hidden-layer=${hidden_layer} --hidden-size=${hidden_size} --init-type=${init_type} \
    --epoch=${epoch} --step-per-collect=${step_per_collect} \
    --buffer-size=${buffer_size} --min-buffer-size=${min_buffer_size} \
    --config ${config} \
    > ~/logs/${alg_type}_${task}_${tag}_3.out 2> ~/logs/${alg_type}_${task}_${tag}_3.err &
    echo "run $seed $tag"
    let seed=$seed+1
    sleep ${time}
done

# ps -ef | grep MountainCar | awk '{print $2}'| xargs kill -9
