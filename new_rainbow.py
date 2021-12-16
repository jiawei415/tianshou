import argparse
import os
import math
import time
import json
import pickle
import pprint
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import NewRainbowPolicy, NewDQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, import_module_or_data, read_config_dict
from tianshou.utils.net.common import NewNet
from tianshou.utils.net.discrete import NewNoisyLinear, NewHyperLinear, EnsembleLinear


def trunc_normal_init(module):
    classname = module.__class__.__name__
    if classname == "Linear":
        bound = 1.0 / math.sqrt(module.in_features)
        nn.init.trunc_normal_(module.weight, std=bound, a=-2*bound, b=2*bound)
        nn.init.zeros_(module.bias)


def xavier_uniform_init(module):
    classname = module.__class__.__name__
    if classname == "Linear":
        gain = 1.0
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def xavier_normal_init(module):
    classname = module.__class__.__name__
    if classname == "Linear":
        gain = 1.0
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


def init_model(model, method='uniform', bias=0.):
    if method == 'xavier':
        init_fn = nn.init.xavier_normal_
    elif method == 'uniform':
        init_fn = nn.init.xavier_uniform_
    else:
        raise NotImplementedError
    for name, param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant(param, bias)
        if 'basedmodel' in name or 'priormodel.model' in name:
            init_fn(param)


def make_env(env_name, max_step=None, size=10, seed=2021):
    env_config = {'size': size, 'seed':seed, 'mapping_seed': seed} if 'DeepSea' in env_name else {}
    env = gym.make(env_name, **env_config)
    if max_step is not None:
        env._max_episode_steps = max_step
    return env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='MountainCar-v0')
    parser.add_argument('--max-step', type=int, default=500)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--eps-test', type=float, default=0.)
    parser.add_argument('--eps-train', type=float, default=0.)
    parser.add_argument('--buffer-size', type=int, default=int(2e5))
    parser.add_argument('--min-replay-size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--hyper-reg-coef', type=float, default=0.01)
    parser.add_argument('--hyper-weight-decay', type=float, default=0.0003125)
    parser.add_argument('--based-weight-decay', type=float, default=0.0003125)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-max', type=float, default=100.)
    parser.add_argument('--target-noise-std', type=float, default=0.)
    parser.add_argument('--prior-scale', type=float, default=10.)
    parser.add_argument('--prior-std', type=float, default=0.)
    parser.add_argument('--noise-std', type=float, default=1.)
    parser.add_argument('--noise-dim', type=int, default=0)
    parser.add_argument('--noisy-std', type=float, default=0.1)
    parser.add_argument('--ensemble-num', type=int, default=0)
    parser.add_argument('--ensemble-sizes', type=int, nargs='*', default=[])
    parser.add_argument('--use-dueling', action="store_true", default=True)
    parser.add_argument('--init-type', type=str, default=None, help="trunc_normal, xavier_uniform, xavier_normal")
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--target-update-freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=2)
    parser.add_argument('--update-per-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--episode-per-test', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='results')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--norm-obs', action="store_true", default=False)
    parser.add_argument('--norm-ret', action="store_true", default=False)
    parser.add_argument('--prioritized', action="store_true", default=False)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--resume-path', type=str, default='')
    parser.add_argument('--evaluation', action="store_true", default=False)
    parser.add_argument('--policy-path', type=str, default='')
    parser.add_argument('--save-interval', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample-per-step', action="store_true", default=False)
    parser.add_argument('--same-noise-update', action="store_true", default=True)
    parser.add_argument('--batch-noise-update', action="store_true", default=True)
    parser.add_argument('--action-sample-num', type=int, default=0)
    parser.add_argument('--action-select-scheme', type=str, default='MAX', help='MAX, VIDS')
    parser.add_argument('--value-gap-eps', type=float, default=1e-3)
    parser.add_argument('--value-var-eps', type=float, default=1e-3)
    parser.add_argument('--config', type=str, default="{}",
                        help="game config eg., {'seed':2021,'size':20,'hidden_sizes':[128,128],'ensemble_num':4,'noise_dim':2,'prior_std':2,'hyper_reg_coef':0.01,}")
    args = parser.parse_known_args()[0]
    return args


def main(args=get_args()):
    # environment
    def make_thunk(seed):
        return lambda: make_env(
            env_name=args.task,
            max_step=args.max_step,
            size=args.size,
            seed=seed,
        )
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv([make_thunk(seed=args.seed)], norm_obs=args.norm_obs)
    test_envs = DummyVectorEnv([make_thunk(seed=args.seed)], norm_obs=args.norm_obs)
    args.min_replay_size = args.max_step
    if 'DeepSea' in args.task:
        train_action_mappling = np.array([action_mapping() for action_mapping in train_envs.get_action_mapping])
        test_action_mappling = np.array([action_mapping() for action_mapping in test_envs.get_action_mapping])
        assert (train_action_mappling == test_action_mappling).all()
        args.min_replay_size = args.size
    args.state_shape = train_envs.observation_space[0].shape or train_envs.observation_space[0].n
    args.action_shape = train_envs.action_space[0].shape or train_envs.action_space[0].n

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    def last_linear(x, y, device):
        if args.ensemble_num:
            return EnsembleLinear(x, y, device, ensemble_num=args.ensemble_num, ensemble_sizes=args.ensemble_sizes, prior_std=args.prior_std, prior_scale=args.prior_scale)
        elif args.noise_dim:
            return NewHyperLinear(x, y, device, noise_dim=args.noise_dim, prior_std=args.prior_std, prior_scale=args.prior_scale, batch_noise=args.batch_noise_update)
        elif args.noisy_std:
            return NewNoisyLinear(x, y, device, noisy_std=args.noisy_std, prior_std=args.prior_std, prior_scale=args.prior_scale, batch_noise=args.batch_noise_update)
        else:
            NotImplementedError

    model_params = {
        "state_shape": args.state_shape,
        "action_shape": args.action_shape,
        "hidden_sizes": args.hidden_sizes,
        "device": args.device,
        "softmax": True,
        "num_atoms": args.num_atoms,
        "prior_std": args.prior_std,
        "ensemble_num": args.ensemble_num,
    }
    if args.use_dueling:
        model_params['last_layer'] = ({ "linear_layer": last_linear}, {"linear_layer": last_linear})
    else:
        model_params['last_layer'] = ({ "linear_layer": last_linear})
    model = NewNet(**model_params).to(args.device)

    if args.init_type == "trunc_normal":
        model.apply(trunc_normal_init)
    elif args.init_type == "xavier_uniform":
        model.apply(xavier_uniform_init)
    elif args.init_type == "xavier_normal":
        model.apply(xavier_normal_init)

    # init_model(model)
    print(f"Network structure:\n{str(model)}")

    # optimizer
    if args.hyper_reg_coef:
        args.hyper_weight_decay = 0
    trainable_params = [
            {'params': (p for name, p in model.named_parameters() if 'priormodel' not in name and 'hypermodel' not in name), 'weight_decay': args.based_weight_decay},
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
        "ensemble_num": args.ensemble_num,
        "noise_std": args.noise_std,
        "noise_dim": args.noise_dim,
        "hyper_reg_coef": hyper_reg_coef,
        "sample_per_step": args.sample_per_step,
        "same_noise_update": args.same_noise_update,
        "batch_noise_update": args.batch_noise_update,
        "target_noise_std": args.target_noise_std,
        "action_sample_num": args.action_sample_num,
        "action_select_scheme": args.action_select_scheme,
    }
    if args.num_atoms > 1:
        policy_params.update({"num_atoms": args.num_atoms, "v_min": -args.v_max, "v_max": args.v_max,})
        policy = NewRainbowPolicy(**policy_params).to(args.device)
    else:
        policy = NewDQNPolicy(**policy_params).to(args.device)

    if args.evaluation:
        policy_name = f"{args.task[:-3].lower()}_{args.seed}_{args.policy_path}"
        policy_path =  os.path.join(args.logdir, "newrainbow", args.task, policy_name, 'policy.pth')
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
    target_noise_dim = args.noise_dim * 2 if model_params.get("dueling_param") else args.noise_dim
    train_collector = Collector(
        policy,
        train_envs,
        buf,
        exploration_noise=False,
        target_noise_dim=target_noise_dim,
        target_noise_std=args.target_noise_std,
        ensemble_num=args.ensemble_num
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.min_replay_size, random=True)

    # log
    if args.ensemble_num:
        alg_type = "ensemble"
    elif args.noise_dim:
        alg_type = "hyper"
    elif args.noisy_std:
        alg_type = "noisy"
    log_name = f"{args.task[:-3].lower()}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.logdir, args.task, alg_type, log_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)
    with open(os.path.join(log_path, "config.json"), "wt") as f:
        kvs = vars(args)
        kvs.pop('config')
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
        # pickle.dump(
        #     train_collector.buffer,
        #     open(os.path.join(log_path, 'train_buffer.pkl'), "wb")
        # )

    if args.resume:
        # load from existing checkpoint
        resume_name = f"{args.task[:-3].lower()}_{args.seed}_{args.resume_path}"
        resume_path =  os.path.join(args.logdir, "newrainbow", args.task, resume_name)
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
    env_name = args.task[:-3].lower()
    config = read_config_dict(args.config)
    alg_config = import_module_or_data(f"tianshou.config.{env_name}_config.alg_config")
    alg_config.update(config)
    for k, v in alg_config.items():
        if k not in args.__dict__.keys():
            print(f'unrecognized config k: {k}, v: {v}, ignored')
            continue
        args.__dict__[k] = v
    main(args)
