import argparse
import os
import time
import json
import pickle
import pprint

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import RainbowPolicy, HyperRainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import NetWithPrior, HyperNetWithPrior
from tianshou.utils.net.discrete import NoisyLinearWithPrior, HyperLinearWithPrior


class NoiseWrapper(gym.Wrapper):
    def __init__(self, env, noise_dim, noise_std=1.):
        super().__init__(env)
        assert noise_dim > 0
        self.env = env
        self.noise_dim = noise_dim
        self.noise_std = noise_std

    def reset(self):
        state = self.env.reset()
        self.now_noise = np.random.normal(0, 1, self.noise_dim) * self.noise_std
        return np.hstack([self.now_noise, state])

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return np.hstack([self.now_noise, state]), reward, done, info


def make_env(env_name, noise_dim=0):
    env = gym.make(env_name)
    env._max_episode_steps = 500
    if noise_dim:
        env = NoiseWrapper(env, noise_dim=noise_dim)
    return env


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Acrobot-v1')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=1e6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hyper-reg-coef', type=float, default=0.01)
    parser.add_argument('--hyper-weight-decay', type=float, default=1e-4)
    parser.add_argument('--base-weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--num-atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-200.)
    parser.add_argument('--v-max', type=float, default=200.)
    parser.add_argument('--prior-std', type=float, default=2.)
    parser.add_argument('--noise-std', type=float, default=1.)
    parser.add_argument('--noise-dim', type=int, default=2)
    parser.add_argument('--noisy-std', type=float, default=0.1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--step-per-epoch', type=int, default=500)
    parser.add_argument('--step-per-collect', type=int, default=8)
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[512, 512])
    parser.add_argument('--training-num', type=int, default=8)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='results')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--prioritized-replay', action="store_true", default=True)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta-final', type=float, default=1.)
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--resume-path', type=str, default='')
    parser.add_argument('--evaluation', action="store_true", default=False)
    parser.add_argument('--policy-path', type=str, default='')
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument('--config', type=str, default="{}",
                        help="game config eg., {'seed':2021,'noise_dim':2,'prior_std':2,'hyper_reg_coef':0.01,}")
    args = parser.parse_known_args()[0]
    return args


def test_rainbow(args=get_args()):
    # environment
    env = make_env(args.task, noise_dim=args.noise_dim)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    env.close()
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = DummyVectorEnv(
        [lambda: make_env(args.task, noise_dim=args.noise_dim) for _ in range(args.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: make_env(args.task, noise_dim=args.noise_dim) for _ in range(args.test_num)]
    )

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # model
    def last_linear(x, y):
        if args.noise_dim:
            return HyperLinearWithPrior(x, y, noize_dim=args.noise_dim, prior_std=args.prior_std)
        else:
            return NoisyLinearWithPrior(x, y, noisy_std=args.noisy_std, prior_std=args.prior_std)

    model_params = {
        "state_shape": args.state_shape,
        "action_shape": args.action_shape,
        "hidden_sizes": args.hidden_sizes,
        "device": args.device,
        "softmax": True,
        "num_atoms": args.num_atoms,
        "prior_std": args.prior_std,
        "dueling_param": ({ "linear_layer": last_linear}, {"linear_layer": last_linear})
    }
    if args.noise_dim:
        model_params.update({"noise_dim": args.noise_dim,})
        model = HyperNetWithPrior(**model_params)
    else:
        model = NetWithPrior(**model_params)
    print(f"Network structure:\n{str(model)}")

    # optimizer
    if args.hyper_reg_coef:
        args.hyper_weight_decay = 0
    trainable_params = [
            {'params': (p for name, p in model.named_parameters() if 'priormodel' not in name and 'hypermodel' not in name),'weight_decay': args.base_weight_decay},
            {'params': (p for name, p in model.named_parameters() if 'priormodel' not in name and 'hypermodel' in name), 'weight_decay': args.hyper_weight_decay},
        ]
    optim = torch.optim.Adam(trainable_params, lr=args.lr)

    # policy
    policy_params = {
        "model": model,
        "optim": optim,
        "discount_factor": args.gamma,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "estimation_step": args.n_step,
        "target_update_freq": args.target_update_freq
    }
    if args.noise_dim:
        hyper_reg_coef = args.hyper_reg_coef / (args.prior_std ** 2) if args.prior_std else args.hyper_reg_coef
        policy_params.update(
            {
                "noise_std": args.noise_std,
                "noise_dim": args.noise_dim,
                "hyper_reg_coef": hyper_reg_coef,
            }
        )
        policy = HyperRainbowPolicy(**policy_params).to(args.device)
    else:
        policy = RainbowPolicy(**policy_params).to(args.device)

    if args.evaluation:
        policy_name = f"{args.task[:-3].lower()}_{args.seed}_{args.policy_path}"
        policy_path =  os.path.join(args.logdir, "hypermodel", args.task, policy_name, 'policy.pth')
        print(f"Loading policy under {policy_path}")
        if os.path.exists(policy_path):
            model = torch.load(policy_path, map_location=args.device)
            policy.load_state_dict(model)
            print("Successfully restore policy.")
        else:
            print("Fail to restore policy.")
        env = make_env(args.task, noise_dim=args.noise_dim)
        policy.eval()
        policy.set_eps(args.eps_test)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")
        return None

    # buffer
    if args.prioritized_replay:
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
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)

    # log
    log_name = f"{args.task[:-3].lower()}_{args.seed}_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.logdir, "hypermodel", args.task, log_name)
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
        if args.prioritized_replay:
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
        pickle.dump(
            train_collector.buffer,
            open(os.path.join(log_path, 'train_buffer.pkl'), "wb")
        )

    if args.resume:
        # load from existing checkpoint
        resume_name = f"{args.task[:-3].lower()}_{args.seed}_{args.resume_path}"
        resume_path =  os.path.join(args.logdir, "hypermodel", args.task, resume_name)
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
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
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


def test_rainbow_resume(args=get_args()):
    args.resume = True
    test_rainbow(args)


def test_prainbow(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 1
    test_rainbow(args)


def test_hyperrainbow(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 2021
    args.hyper_reg_coef = 0.01
    args.prior_std = 2.
    args.noise_std = 1.
    args.noise_dim = 2
    args.noisy_std = 0.1
    test_rainbow(args)


def train_hyperrainbow(args=get_args()):
    args.prioritized_replay = True
    args.gamma = .95
    args.seed = 2021
    args.hyper_reg_coef = 0.01
    args.prior_std = 2.
    args.noise_std = 1.
    args.noise_dim = 2
    args.noisy_std = 0.1
    test_rainbow(args)


if __name__ == '__main__':
    args = get_args()
    config = eval(args.config)
    for k, v in config.items():
        if k not in args.__dict__.keys():
            print(f'unrecognized config k: {k}, v: {v}, ignored')
            continue
        args.__dict__[k] = v
    test_rainbow(args)
    # test_hyperrainbow()