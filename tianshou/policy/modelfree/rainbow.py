import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Optional, Union, Callable


from tianshou.policy import BasePolicy, C51Policy, HyperC51Policy
from tianshou.policy.base import _nstep_return
from tianshou.utils.net.discrete import sample_noise, noisy_layer_noise, hyper_layer_noise
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

class RainbowPolicy(C51Policy):
    """Implementation of Rainbow DQN. arXiv:1710.02298.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int num_atoms: the number of atoms in the support set of the
        value distribution. Default to 51.
    :param float v_min: the value of the smallest atom in the support set.
        Default to -10.0.
    :param float v_max: the value of the largest atom in the support set.
        Default to 10.0.
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.C51Policy` for more detailed
        explanation.
    """

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        sample_noise(self.model)
        if self._target and sample_noise(self.model_old):
            self.model_old.train()  # so that NoisyLinear takes effect
        return super().learn(batch, **kwargs)


class HyperRainbowPolicy(HyperC51Policy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        noise_std: int = 1,
        noise_dim: int = 0,
        num_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        hyper_reg_coef: float = 0.1,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any
    ) -> None:
        super(HyperRainbowPolicy, self).__init__(
            model, optim, discount_factor, num_atoms, v_min, v_max, hyper_reg_coef,
            estimation_step, target_update_freq, reward_normalization, **kwargs
        )
        self.noise_dim = noise_dim
        self.noise_std = noise_std

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        batch_size = batch['obs'].shape[0]
        noise = np.random.normal(0, 1, [batch_size, self.noise_dim]) * self.noise_std
        batch['obs'][:, :self.noise_dim] = noise
        batch['obs_next'][:, :self.noise_dim] = noise
        if self._target:
            self.model_old.train()
        return super().learn(batch, **kwargs)


class NewRainbowPolicy(C51Policy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        num_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        estimation_step: int = 1,
        target_update_freq: int = 100,
        reward_normalization: bool = False,
        noise_dim: int = 0,
        noise_std: float = 1.,
        hyper_reg_coef: float = 0.001,
        ensemble_num: int = 0,
        sample_per_step: bool = False,
        same_noise_update: bool = True,
        batch_noise: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model, optim, discount_factor=discount_factor, num_atoms=num_atoms, v_min=v_min, v_max=v_max,
            estimation_step=estimation_step, target_update_freq=target_update_freq, reward_normalization=reward_normalization,
        **kwargs)
        self.last_layer_inp_dim = model.basedmodel.output_dim
        self.v_out_dim = model.V.output_dim
        self.q_out_dim = model.Q.output_dim
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.hyper_reg_coef = hyper_reg_coef
        self.sample_per_step = sample_per_step
        self.same_noise_update = same_noise_update
        self.batch_noise = batch_noise
        self.ensemble_num = ensemble_num
        self.active_head_train = None
        self.active_head_test = None

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        if self.ensemble_num:
            return self.support.repeat(len(indices), self.ensemble_num, 1)
        return self.support.repeat(len(indices), 1)

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        logits = (logits * self.support).sum(-1)
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def _target_dist(self, batch: Batch, noise: Dict[str, Any] = {}) -> torch.Tensor:
        main_noise  = noise if self.same_noise_update else self.reset_noise(batch['obs'].shape[0], reset=False)
        target_noise = noise if self.same_noise_update else self.reset_noise(batch['obs'].shape[0], reset=False)
        with torch.no_grad():
            a = self(batch, input="obs_next", is_collecting=False, noise=main_noise).act
            next_dist = self(batch, model="model_old", input="obs_next", is_collecting=False, noise=target_noise).logits
        support = self.support.view(1, -1, 1)
        target_support = batch.returns.clamp(self._v_min, self._v_max).unsqueeze(-2)
        if self.ensemble_num:
            a_one_hot = F.one_hot(torch.as_tensor(a, device=next_dist.device), self.max_action_num).to(torch.float32)
            next_dist = torch.einsum('bkat,bka->bkt', next_dist, a_one_hot)
            support = support.unsqueeze(1).repeat(1, self.ensemble_num, 1, 1)
        else:
            next_dist = next_dist[np.arange(len(a)), a, :]
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (1 - (target_support - support).abs() /self.delta_z).clamp(0, 1) * next_dist.unsqueeze(-2)
        return target_dist.sum(-1)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        noise: Dict[str, Any] = {},
        is_collecting: bool = True,
        **kwargs: Any
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        done = batch['done'][0] if len(batch['done'].shape) > 0 else True
        if self.ensemble_num:
            if is_collecting:
                if self.training:
                    if self.active_head_train is None or done:
                        self.active_head_train = np.random.randint(low=0, high=self.ensemble_num)
                    logits, h = model(obs_, state=state, active_head=self.active_head_train, info=batch.info)
                else:
                    if self.active_head_test is None or done:
                        self.active_head_test = np.random.randint(low=0, high=self.ensemble_num)
                    logits, h = model(obs_, state=state, active_head=self.active_head_test, info=batch.info)
            else:
                logits, h = model(obs_, state=state, active_head=None, info=batch.info)
        else:
            if is_collecting and self.sample_per_step:
                self.reset_noise(obs_.shape[0], reset=True)
            elif is_collecting and done:
                self.reset_noise(obs_.shape[0], reset=True)
            logits, h = model(obs_, state=state, info=batch.info, noise=noise)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        act = to_numpy(q.max(dim=-1)[1])
        return Batch(logits=logits, act=act, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        if self.ensemble_num:
            target_dist = self._target_dist(batch)
            curr_dist = self(batch, is_collecting=False).logits
            act = batch.act
            act_one_hot = F.one_hot(torch.as_tensor(act, device=curr_dist.device), self.max_action_num).to(torch.float32)
            curr_dist = torch.einsum('bkat,ba->bkt', curr_dist, act_one_hot) # (None, ensemble_num, num_atoms)
            masks = to_torch_as(batch.ensemble_mask, curr_dist)
            cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(-1)
            cross_entropy *= masks
        else:
            batch_size = batch['obs'].shape[0] if self.batch_noise else 1
            noise = self.reset_noise(batch_size, reset=False)
            target_dist = self._target_dist(batch, noise)
            curr_dist = self(batch, is_collecting=False, noise=noise).logits
            act = batch.act
            curr_dist = curr_dist[np.arange(len(act)), act, :]
            cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(-1)
        weight = batch.pop("weight", 1.0)
        loss = (cross_entropy * weight).mean(0).sum()
        if self.hyper_reg_coef and self.noise_dim:
            reg_loss = self.model.Q.model.regularization(noise['Q']) + self.model.V.model.regularization(noise['V'])
            loss += reg_loss * (self.hyper_reg_coef / kwargs['sample_num'])
        batch.weight = cross_entropy.detach()  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def reset_noise(self, batch_size: int = 1, reset: bool = True):
        if self.noise_dim:
            hyper_noise = hyper_layer_noise(batch_size, self.noise_dim * 2, self.noise_std)
            Q_noise, V_noise = hyper_noise.split([self.noise_dim, self.noise_dim], dim=1)
            noise = {'Q': {'hyper_noise': Q_noise}, 'V': {'hyper_noise': V_noise}}
        else:
            Q_eps_p, Q_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.q_out_dim)
            V_eps_p, V_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.v_out_dim)
            noise = {'Q': {'eps_p': Q_eps_p, 'eps_q': Q_eps_q}, 'V': {'eps_p': V_eps_p, 'eps_q': V_eps_q}}
        if reset:
            self.model.Q.model.reset_noise(noise['Q'])
            self.model.V.model.reset_noise(noise['V'])
        return noise

    @staticmethod
    def compute_nstep_return(
        batch: Batch,
        buffer: ReplayBuffer,
        indice: np.ndarray,
        target_q_fn: Callable[[ReplayBuffer, np.ndarray], torch.Tensor],
        gamma: float = 0.99,
        n_step: int = 1,
        rew_norm: bool = False,
    ) -> Batch:
        assert not rew_norm, \
            "Reward normalization in computing n-step returns is unsupported now."
        rew = buffer.rew
        bsz = len(indice)
        indices = [indice]
        for _ in range(n_step - 1):
            indices.append(buffer.next(indices[-1]))
        indices = np.stack(indices)
        # terminal indicates buffer indexes nstep after 'indice',
        # and are truncated at the end of each episode
        terminal = indices[-1]

        with torch.no_grad():
            target_q_torch = target_q_fn(buffer, terminal)
        if len(target_q_torch.shape) > 2:
            # Ziniu Li: return contains all target_q for ensembles
            end_flag = buffer.done.copy()
            end_flag[buffer.unfinished_index()] = True
            target_qs = []
            for k in range(target_q_torch.shape[1]):
                target_q = to_numpy(target_q_torch[:, k].reshape(bsz, -1))
                target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
                target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
                target_qs.append(target_q)
            target_qs = np.array(target_qs)   # (ensemble_num, None, num_atoms)
            target_qs = target_qs.transpose([1, 0, 2])  # (None, ensemble_num, num_atoms)
            batch.returns = to_torch_as(target_qs, target_q_torch)
        else:
            target_q = to_numpy(target_q_torch.reshape(bsz, -1))
            target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
            end_flag = buffer.done.copy()
            end_flag[buffer.unfinished_index()] = True
            target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
            batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch
