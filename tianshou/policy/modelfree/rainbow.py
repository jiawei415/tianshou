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
        action_sample_num: int = 0,
        action_select_scheme: str = None,
        value_var_eps: float = 1e-3,
        value_gap_eps: float = 1e-3,
        sample_per_step: bool = False,
        target_noise_std: float = 0.,
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
        self.target_noise_std = target_noise_std
        self.hyper_reg_coef = hyper_reg_coef
        self.sample_per_step = sample_per_step
        self.same_noise_update = same_noise_update
        self.batch_noise = batch_noise
        self.noise_train = None
        self.noise_test = None
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.ensemble_num = ensemble_num
        self.active_head_train = None
        self.active_head_test = None
        self.value_gap_eps = value_gap_eps
        self.value_var_eps = value_var_eps

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
        main_model_noise  = noise if self.same_noise_update else self.sample_noise(batch['obs'].shape[0])
        target_model_noise = noise if self.same_noise_update else self.sample_noise(batch['obs'].shape[0])
        with torch.no_grad():
            a = self(batch, input="obs_next", noise=main_model_noise).act # (None,) or (None, ensemble_num)
            next_dist = self(batch, model="model_old", input="obs_next", noise=target_model_noise).logits # (None, action_num, num_atoms) or (None, ensemble_num, action_num, num_atoms)
        support = self.support.view(1, -1, 1) # (1, num_atoms, 1)
        target_support = batch.returns.clamp(self._v_min, self._v_max).unsqueeze(-2) # (None, 1, num_atoms) or (None, ensemble_num, 1, num_atoms)
        if self.ensemble_num:
            a_one_hot = F.one_hot(torch.as_tensor(a, device=next_dist.device), self.max_action_num).to(torch.float32) # (None, ensemble_num, action_num)
            next_dist = torch.einsum('bkat,bka->bkt', next_dist, a_one_hot) # (None, ensemble_num, num_atoms)
            support = support.unsqueeze(1).repeat(1, self.ensemble_num, 1, 1) # (1, ensemble_num, num_atoms, 1)
        else:
            next_dist = next_dist[np.arange(len(a)), a, :] # (None, num_atoms)
            if self.target_noise_std and self.noise_dim:
                update_noise = torch.cat([target_model_noise['Q']['hyper_noise'], target_model_noise['V']['hyper_noise']], dim=1)
                target_noise = torch.tensor(batch.target_noise)
                loss_noise = torch.sum(target_noise.mul(update_noise).to(next_dist.device), dim=1, keepdim=True)
                next_dist += loss_noise
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (1 - (target_support - support).abs() / self.delta_z).clamp(0, 1) * next_dist.unsqueeze(-2) # (None, num_atoms, num_atoms) or (None, ensemble_num, num_atoms, num_atoms)
        return target_dist.sum(-1) # (None, num_atoms) or (None, ensemble_num, num_atoms)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        active_head: Any = None,
        noise: Dict[str, Any] = {},
        **kwargs: Any
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        done = batch['done'][0] if len(batch['done'].shape) > 0 else True
        if self.ensemble_num:
            if not self.updating:
                if not self.action_sample_num:
                    if self.training:
                        if self.active_head_train is None or done:
                            self.active_head_train = np.random.randint(low=0, high=self.ensemble_num)
                        active_head = self.active_head_train
                    else:
                        if self.active_head_test is None or done:
                            self.active_head_test = np.random.randint(low=0, high=self.ensemble_num)
                        active_head = self.active_head_test
            logits, h = model(obs_, state=state, active_head=active_head, info=batch.info) # (None, ensemble_num, num_atoms)
        else:
            if not self.updating:
                if self.action_sample_num:
                    noise_num = self.action_sample_num
                    obs_ = obs_.repeat(self.action_sample_num, axis=0)
                else:
                    noise_num = obs_.shape[0]
                if self.training:
                    if self.noise_train is None or self.sample_per_step or done:
                        self.noise_train = self.sample_noise(noise_num)
                    noise = self.noise_train
                else:
                    if self.noise_test is None or self.sample_per_step or done:
                        self.noise_test = self.sample_noise(noise_num)
                    noise = self.noise_test
            logits, h = model(obs_, state=state, noise=noise, info=batch.info) # (None, num_atoms)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None,) or (None, ensemble_num)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        if self.action_sample_num and not self.updating:
            if self.action_select_scheme == "MAX":
                act = to_numpy(torch.argmax(q.squeeze(0)) % self.max_action_num).reshape(1)
            elif self.action_select_scheme == "VIDS":
                value_gap = q.max(dim=-1, keepdim=True)[0] - q
                value_gap = value_gap.mean(dim=0) + self.value_gap_eps
                value_var = torch.var(logits, dim=0)
                value_var = value_var.mean(dim=1) + self.value_var_eps
                act = to_numpy(torch.argmin(value_gap / value_var)).reshape(1)
            else:
                raise ValueError(self.action_select_scheme)
        else:
            act = to_numpy(q.max(dim=-1)[1])
        return Batch(logits=logits, act=act, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        if self.ensemble_num:
            target_dist = self._target_dist(batch) # (None, ensemble_num, num_atoms)
            curr_dist = self(batch).logits # (None, ensemble_num, action_num, num_atoms)
            act = batch.act # (None,)
            act_one_hot = F.one_hot(torch.as_tensor(act, device=curr_dist.device), self.max_action_num).to(torch.float32) # (None, action_num)
            curr_dist = torch.einsum('bkat,ba->bkt', curr_dist, act_one_hot) # (None, ensemble_num, num_atoms)
            masks = to_torch_as(batch.ensemble_mask, curr_dist) # (None, ensemble_num)
            cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(-1) # (None, ensemble_num)
            cross_entropy *= masks # (None, ensemble_num)
        else:
            noise_num = batch['obs'].shape[0] if self.batch_noise else 1
            update_noise = self.sample_noise(noise_num)
            target_dist = self._target_dist(batch, noise=update_noise) # (None, num_atoms)
            curr_dist = self(batch, noise=update_noise).logits # (None, action_num, num_atoms)
            act = batch.act # (None,)
            curr_dist = curr_dist[np.arange(len(act)), act, :] # (None, num_atoms)
            cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(-1) # (None,)
        weight = batch.pop("weight", 1.0)
        loss = (cross_entropy * weight).mean(0).sum()
        if self.hyper_reg_coef and self.noise_dim:
            reg_loss = self.model.Q.model.regularization(update_noise['Q']) + self.model.V.model.regularization(update_noise['V'])
            loss += reg_loss * (self.hyper_reg_coef / kwargs['sample_num'])
        batch.weight = cross_entropy.detach()  # prio-buffer
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def sample_noise(self, batch_size: int = 1, reset: bool = False):
        if self.noise_dim:
            hyper_noise = hyper_layer_noise(batch_size, self.noise_dim * 2, self.noise_std)
            Q_noise, V_noise = hyper_noise.split([self.noise_dim, self.noise_dim], dim=1)
            noise = {'Q': {'hyper_noise': Q_noise}, 'V': {'hyper_noise': V_noise}}
        else:
            Q_eps_p, Q_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.q_out_dim)
            V_eps_p, V_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.v_out_dim)
            noise = {'Q': {'eps_p': Q_eps_p, 'eps_q': Q_eps_q}, 'V': {'eps_p': V_eps_p, 'eps_q': V_eps_q}}
        # if reset:
        #     self.model.Q.model.reset_noise(noise['Q'])
        #     self.model.V.model.reset_noise(noise['V'])
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
            target_q_torch = target_q_fn(buffer, terminal) # (Nonr, num_atoms) or (Nonr, ensemble_num, num_atoms)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        if len(target_q_torch.shape) > 2:
            # Ziniu Li: return contains all target_q for ensembles
            target_qs = []
            for k in range(target_q_torch.shape[1]):
                target_q = to_numpy(target_q_torch[:, k].reshape(bsz, -1)) # (Nonr, num_atoms)
                target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1) # (Nonr, num_atoms)
                target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step) # (Nonr, num_atoms)
                target_qs.append(target_q)
            target_qs = np.array(target_qs) # (ensemble_num, None, num_atoms)
            target_qs = target_qs.transpose([1, 0, 2]) # (None, ensemble_num, num_atoms)
            batch.returns = to_torch_as(target_qs, target_q_torch)
        else:
            target_q = to_numpy(target_q_torch.reshape(bsz, -1)) # (Nonr, num_atoms)
            target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1) # (Nonr, num_atoms)
            target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step) # (Nonr, num_atoms)
            batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch
