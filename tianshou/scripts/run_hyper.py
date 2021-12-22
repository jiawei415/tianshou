import argparse
import os
import time
import json
import pickle
import pprint
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.utils import make_env
from tianshou.policy import  HyperDQNPolicy, HyperC51Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, import_module_or_data, read_config_dict
from tianshou.utils.net.common import HyperNet, trunc_normal_init, xavier_normal_init, xavier_uniform_init
from tianshou.utils.net.discrete import NewHyperLinear, MultiHyperLinear


def get_args():
    parser = argparse.ArgumentParser()
    # environment config
    parser.add_argument('--task', type=str, default='MountainCar-v0')
    parser.add_argument('--max-step', type=int, default=500)
    parser.add_argument('--size', type=int, default=20, help="only for DeepSea-v0")
    parser.add_argument('--length', type=int, default=20, help="only for CustomizeMDP-v1/v2")
    parser.add_argument('--final-reward', type=int, default=2, help="only for CustomizeMDP-v1/v2. If it is not 0 or 1, it means randomly generated")
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--norm-obs', action="store_true", default=False)
    parser.add_argument('--norm-ret', action="store_true", default=False)
    # training config
    parser.add_argument('--same-noise-update', action="store_true", default=True)
    parser.add_argument('--batch-noise-update', action="store_true", default=True)
    parser.add_argument('--target-update-freq', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.0003125)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--v-max', type=float, default=100.)
    parser.add_argument('--num-atoms', type=int, default=51)
    # algorithm config
    parser.add_argument('--alg-type', type=str, default="hyper")
    parser.add_argument('--noise-std', type=float, default=1.)
    parser.add_argument('--noise-dim', type=int, default=2, help="Greater than 0 means using HyperModel")
    parser.add_argument('--prior-std', type=float, default=1., help="Greater than 0 means using priormodel")
    parser.add_argument('--prior-scale', type=float, default=10.)
    parser.add_argument('--target-noise-std', type=float, default=0.)
    parser.add_argument('--hyper-reg-coef', type=float, default=0.01)
    parser.add_argument('--hyper-weight-decay', type=float, default=0.0003125)
    # network config
    parser.add_argument('--hidden-layer', type=int, default=2)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--use-dueling', action="store_true", default=True)
    parser.add_argument('--use-multihyper', type=int, default=1, help="1 means use")
    parser.add_argument('--init-type', type=str, default="", help="trunc_normal, xavier_uniform, xavier_normal")
    # epoch config
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=2)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--episode-per-test', type=int, default=10)
    # buffer confing
    parser.add_argument('--buffer-size', type=int, default=int(2e5))
    parser.add_argument('--min-buffer-size', type=int, default=500)
    parser.add_argument('--prioritized', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    # action selection confing
    parser.add_argument('--eps-test', type=float, default=0.)
    parser.add_argument('--eps-train', type=float, default=0.)
    parser.add_argument('--sample-per-step', action="store_true", default=False)
    parser.add_argument('--action-sample-num', type=int, default=1)
    parser.add_argument('--action-select-scheme', type=str, default="Greedy", help='MAX, VIDS, Greedy')
    parser.add_argument('--value-gap-eps', type=float, default=1e-3)
    parser.add_argument('--value-var-eps', type=float, default=1e-3)
    # other confing
    parser.add_argument('--save-interval', type=int, default=4)
    parser.add_argument('--save-buffer', action="store_true", default=False)
    parser.add_argument('--logdir', type=str, default='~/results/tianshou')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--resume-path', type=str, default='')
    parser.add_argument('--evaluation', action="store_true", default=False)
    parser.add_argument('--policy-path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # overwrite config
    parser.add_argument('--config', type=str, default="{}",
                        help="game config eg., {'seed':2021,'size':20,'hidden_size':128,'noise_dim':2,'prior_std':0,'num_atoms':51}")
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # environment
    if args.task.startswith("DeepSea"):
        env_config = {'seed':args.seed, 'size': args.size, 'mapping_seed': args.seed}
    elif args.task.startswith("CustomizeMDP"):
        env_config = {'seed':args.seed, 'length': args.length, 'final_reward': args.final_reward}
    else:
        env_config = {}
    def make_thunk():
        return lambda: make_env(env_name=args.task, max_step=args.max_step, env_config=env_config)
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([make_thunk()], norm_obs=args.norm_obs)
    test_envs = DummyVectorEnv([make_thunk()], norm_obs=args.norm_obs)
    if args.task.startswith('DeepSea') or args.task.startswith('CustomizeMDP'):
        train_action_mappling = np.array([action_mapping() for action_mapping in train_envs._get_action_mapping])
        test_action_mappling = np.array([action_mapping() for action_mapping in test_envs._get_action_mapping])
        assert (train_action_mappling == test_action_mappling).all()
        args.max_step = args.size
    if args.task.startswith('CustomizeMDP'):
        train_all_rewards = np.array([get_rewards() for get_rewards in train_envs._get_rewards])
        test_all_rewards = np.array([get_rewards() for get_rewards in test_envs._get_rewards])
        assert (train_all_rewards == test_all_rewards).all()
        args.max_step = args.length
    args.state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
    args.action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    last_layer_params = {
        'device': args.device,
        'noise_dim': args.noise_dim,
        'prior_std': args.prior_std,
        'prior_scale': args.prior_scale,
        'batch_noise': args.batch_noise_update,
        'action_num': args.action_shape,
    }
    def q_last_layer(x, y):
        if args.use_multihyper:
            return MultiHyperLinear(x, y, **last_layer_params)
        else:
            return NewHyperLinear(x, y, **last_layer_params)
    def v_last_layer(x, y):
        return NewHyperLinear(x, y, **last_layer_params)

    args.hidden_sizes = [args.hidden_size] * args.hidden_layer
    model_params = {
        "state_shape": args.state_shape,
        "action_shape": args.action_shape,
        "hidden_sizes": args.hidden_sizes,
        "device": args.device,
        "softmax": True,
        "num_atoms": args.num_atoms,
        "prior_std": args.prior_std,
        "use_dueling": args.use_dueling,
    }
    if args.use_dueling:
        model_params['last_layers'] = ({ "last_layer": q_last_layer}, {"last_layer": v_last_layer})
    else:
        model_params['last_layers'] = ({ "last_layer": q_last_layer}, )
    model = HyperNet(**model_params).to(args.device)

    if args.init_type == "trunc_normal":
        model.apply(trunc_normal_init)
    elif args.init_type == "xavier_uniform":
        model.apply(xavier_uniform_init)
    elif args.init_type == "xavier_normal":
        model.apply(xavier_normal_init)

    # init_model(model)
    print(f"Network structure:\n{str(model)}")
    print(f"Network parameters: {sum(param.numel() for param in model.parameters())}")

    # optimizer
    if args.hyper_reg_coef:
        args.hyper_weight_decay = 0
    trainable_params = [
            {'params': (p for name, p in model.named_parameters() if 'priormodel' not in name and 'hypermodel' not in name), 'weight_decay': args.weight_decay},
            {'params': (p for name, p in model.named_parameters() if 'priormodel' not in name and 'hypermodel' in name), 'weight_decay': args.hyper_weight_decay},
        ]
    optim = torch.optim.Adam(trainable_params, lr=args.lr)

    # policy
    hyper_reg_coef = args.hyper_reg_coef / (args.prior_std ** 2) if args.prior_std else args.hyper_reg_coef
    policy_params = {
        "model": model,
        "optim": optim,
        "discount_factor": args.gamma,
        "estimation_step": args.n_step,
        "target_update_freq": args.target_update_freq,
        "reward_normalization": args.norm_ret,
        "use_dueling": args.use_dueling,
        "same_noise_update": args.same_noise_update,
        "batch_noise_update": args.batch_noise_update,
        "sample_per_step": args.sample_per_step,
        "action_sample_num": args.action_sample_num,
        "action_select_scheme": args.action_select_scheme,
        "noise_std": args.noise_std,
        "noise_dim": args.noise_dim,
        "hyper_reg_coef": hyper_reg_coef,
        "use_target_noise": bool(args.target_noise_std)
    }
    if args.num_atoms > 1:
        policy_params.update({'num_atoms': args.num_atoms, 'v_max': args.v_max, 'v_min': -args.v_max})
        policy = HyperC51Policy(**policy_params).to(args.device)
    else:
        policy = HyperDQNPolicy(**policy_params).to(args.device)

    if args.evaluation:
        policy_path =  os.path.join(args.policy_path, 'policy.pth')
        print(f"Loading policy under {policy_path}")
        if os.path.exists(policy_path):
            model = torch.load(policy_path, map_location=args.device)
            policy.load_state_dict(model)
            print("Successfully restore policy.")
        else:
            print("Fail to restore policy.")
        env = DummyVectorEnv([make_thunk(seed=args.seed)])
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        return None

    # buffer
    if args.prioritized:
        buf = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=True
        )
    else:
        buf = VectorReplayBuffer(args.buffer_size, buffer_num=len(train_envs))

    # collector
    target_noise_dim = args.noise_dim * 2 if model.use_dueling else args.noise_dim
    train_collector = Collector(
        policy,
        train_envs,
        buf,
        exploration_noise=False,
        target_noise_dim=target_noise_dim,
        target_noise_std=args.target_noise_std
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    train_collector.collect(n_step=args.min_buffer_size, random=True)

    # log
    log_file = f"{args.task[:-3].lower()}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.join(args.logdir, args.task, args.alg_type.lower(), log_file)
    log_path = os.path.expanduser(log_path)
    os.makedirs(os.path.expanduser(log_path), exist_ok=True)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)
    with open(os.path.join(log_path, "config.json"), "wt") as f:
        kvs = vars(args)
        print(kvs)
        f.write(json.dumps(kvs, indent=4) + '\n')
        f.flush()
        f.close()

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    def train_fn(epoch, env_step):
        # eps annealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)
        # beta annealing, just a demo
        if args.prioritized:
            if env_step <= 10000:
                beta = args.beta
            elif env_step <= 50000:
                beta = args.beta - (env_step - 10000) / \
                    40000 * (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buf.set_beta(beta)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(
            {
                'model': policy.state_dict(),
                'optim': optim.state_dict(),
            }, os.path.join(log_path, 'checkpoint.pth')
        )
        if args.save_buffer:
            pickle.dump(
                train_collector.buffer,
                open(os.path.join(log_path, 'train_buffer.pkl'), "wb")
            )

    if args.resume:
        # load from existing checkpoint
        resume_path =  args.resume_path
        print(f"Loading agent under {resume_path}")
        ckpt_path = os.path.join(resume_path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            policy.optim.load_state_dict(checkpoint['optim'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(resume_path, 'train_buffer.pkl')
        if os.path.exists(buffer_path):
            train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    # trainer
    # args.step_per_collect *= args.training_num
    args.update_per_step /= args.step_per_collect
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        episode_per_test=args.episode_per_test,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=None,
        save_fn=save_fn,
        logger=logger,
        resume_from_log=args.resume,
        save_checkpoint_fn=save_checkpoint_fn
    )
    # assert stop_fn(result['best_reward'])
    pprint.pprint(result)


if __name__ == '__main__':
    args = get_args()
    config = read_config_dict(args.config)
    for k, v in config.items():
        if k not in args.__dict__.keys():
            print(f'unrecognized config k: {k}, v: {v}, ignored')
            continue
        args.__dict__[k] = v
    main(args)
