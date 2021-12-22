export CUDA_VISIBLE_DEVICES=$1
seed=$2

## environment config
task=CustomizeMDP-v1 # v2
length=10
final_reward=2
## training config
same_noise_update=True
batch_noise_update=True
target_update_freq=4
batch_size=128
lr=0.0001
weight_decay=0.0003125
v_max=10
num_atoms=51
## algorithm config
alg_type=noisy
noisy_std=0.1
prior_std=1
prior_scale=10
## action selection config
sample_per_step=False
action_sample_num=1
action_select_scheme=Greedy
value_gap_eps=0.001
value_var_eps=0.001
## network config
hidden_layer=2
hidden_size=64
use_dueling=True
init_type=trunc_normal
## epoch config
epoch=1000
step_per_collect=1
## buffer config
buffer_size=200000
min_buffer_size=${length}

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
    python -m tianshou.scripts.run_${alg_type} --seed ${seed} --task ${task} --length ${length} --final-reward ${final_reward} \
    --target-update-freq=${target_update_freq} --batch-size=${batch_size} --lr=${lr} \
    --weight-decay=${weight_decay} --v-max=${v_max} --num-atoms=${num_atoms} \
    --noisy-std=${noisy_std} --prior-std=${prior_std} --prior-scale=${prior_scale} \
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

# ps -ef | grep CustomizeMDP-v1 | awk '{print $2}'| xargs kill -9
