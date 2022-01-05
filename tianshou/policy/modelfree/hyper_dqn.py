import torch
import numpy as np
from typing import Any, Dict, Optional, Union, Callable

from tianshou.policy import DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


class SampleNoise():
    def __init__(self, use_dueling: bool, noise_dim: int, noise_std: float=1., noise_norm: bool=False):
        self.noise_std = noise_std
        self.noise_dim = noise_dim
        self.noise_norm = noise_norm
        self.use_dueling = use_dueling
        self.sample_noise = self._sample_noise_v1

    def _sample_noise_v1(self, batch_size: int):
        hyper_noise = torch.randn(size=(batch_size, self.noise_dim)) * self.noise_std
        if self.noise_dim > 1 and self.noise_norm:
            hyper_noise = hyper_noise / torch.norm(hyper_noise, dim=1, keepdim=True)
        noise = {'Q': {'hyper_noise': hyper_noise}}
        if self.use_dueling:
            noise.update({'V': {'hyper_noise': hyper_noise}})
        return noise

    def _sample_noise_v2(self, batch_size: int):
        hyper_noise = torch.randn(size=(batch_size, self.noise_dim * 2)) * self.noise_std
        if self.noise_dim > 1 and self.noise_norm:
            hyper_noise = hyper_noise / torch.norm(hyper_noise, dim=1, keepdim=True)
        Q_noise, V_noise = hyper_noise.split([self.noise_dim, self.noise_dim], dim=1)
        noise = {'Q': {'hyper_noise': Q_noise}, 'V': {'hyper_noise': V_noise}}
        return noise

    def _sample_noise_v3(self, batch_size: int):
        hyper_noise = torch.randn(size=(batch_size, self.noise_dim)) * self.noise_std
        if self.noise_dim > 1 and self.noise_norm:
            hyper_noise = hyper_noise / torch.norm(hyper_noise, dim=1, keepdim=True)
        noise = {'Q': {'hyper_noise': hyper_noise}}
        if self.use_dueling:
            hyper_noise = torch.randn(size=(batch_size, self.noise_dim)) * self.noise_std
            if self.noise_dim > 1 and self.noise_norm:
                hyper_noise = hyper_noise / torch.norm(hyper_noise, dim=1, keepdim=True)
            noise.update({'V': {'hyper_noise': hyper_noise}})
        return noise


class HyperDQNPolicy(DQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        use_dueling: bool = True,
        same_noise_update: bool = True,
        batch_noise_update: bool = True,
        sample_per_step: bool = False,
        action_sample_num: int = 1,
        action_select_scheme: str = None,
        value_var_eps: float = 1e-3,
        value_gap_eps: float = 1e-3,
        noise_std: float = 1.,
        noise_dim: int = 2,
        noise_norm: bool = False,
        hyper_reg_coef: float = 0.001,
        use_target_noise: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, **kwargs
        )
        self.use_dueling = use_dueling
        self.same_noise_update = same_noise_update
        self.batch_noise_update = batch_noise_update
        self.sample_per_step = sample_per_step
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.value_var_eps = value_var_eps
        self.value_gap_eps = value_gap_eps
        self.noise_std = noise_std
        self.noise_dim = noise_dim
        self.noise_norm = noise_norm
        self.use_target_noise = use_target_noise
        self.hyper_reg_coef = hyper_reg_coef
        self.noise_test = None
        self.noise_train = None
        self.noise_update = None
        self.sample_noise = SampleNoise(use_dueling, noise_dim, noise_std, noise_norm).sample_noise
        if self.action_select_scheme == "Greedy":
            assert self.action_sample_num == 1
            self.get_actions = getattr(self, '_greedy_action_select')
        elif self.action_select_scheme == 'MAX':
            self.get_actions = getattr(self, '_max_action_select')
        elif self.action_select_scheme == 'VIDS':
            self.get_actions = getattr(self, '_vids_action_select')
        else:
            raise NotImplementedError(f'No action selcet scheme {self.action_select_scheme}')

    def _vids_action_select(self, q):
        value_gap = q.max(dim=-1, keepdim=True)[0] - q
        value_gap = value_gap.mean(dim=0) + self.value_gap_eps
        value_var = torch.var(q, dim=0) + self.value_var_eps
        act = to_numpy(torch.argmin(value_gap / value_var)).reshape(1)
        return act

    def _max_action_select(self, q):
        action_num = q.shape[-1]
        act = to_numpy(torch.argmax(q) % action_num).reshape(1)
        return act

    def _greedy_action_select(self, q):
        act = to_numpy(q.max(dim=-1)[1])
        return act

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        batch_size = len(indices)
        if self.same_noise_update:
            main_model_noise = self.noise_update
            target_model_noise = self.noise_update
        else:
            main_model_noise = self.sample_noise(batch_size)
            target_model_noise = self.sample_noise(batch_size)
        if self._is_double:
            with torch.no_grad():
                a = self(batch, model="model", input="obs_next", noise=main_model_noise).act # (None,)
                target_q = self(batch, model="model_old", input="obs_next", noise=target_model_noise).logits # (None, action_num)
            target_q = target_q[np.arange(len(a)), a]
        else:
            with torch.no_grad():
                target_q = self(batch, model="model_old", input="obs_next", noise=target_model_noise).logits # (None, action_num)
            target_q = torch.max(target_q, dim=-1)[0]
        return target_q

    def forward(
        self, batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        noise: Dict[str, Any] = None,
        **kwargs: Any
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        if not self.updating:
            obs_ = obs_.repeat(self.action_sample_num, axis=0)
            done = len(batch.done.shape) == 0 or batch.done[0]
            if self.training:
                if self.noise_train is None or self.sample_per_step or done:
                    self.noise_train = self.sample_noise(self.action_sample_num)
                noise = self.noise_train
            else:
                if self.noise_test is None or self.sample_per_step or done:
                    self.noise_test = self.sample_noise(self.action_sample_num)
                noise = self.noise_test
            logits, h = model(obs_, state=state, noise=noise, info=batch.info) # (None)
            q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None,)
            act = self.get_actions(q)
        else:
            logits, h = model(obs_, state=state, noise=noise, info=batch.info) # (None)
            q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None,)
            act = to_numpy(q.max(dim=-1)[1])
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        return Batch(logits=logits, act=act, state=h)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        kwargs.update({"sample_num": len(buffer)})
        batch, indices = buffer.sample(sample_size)
        self.noise_update = self.sample_noise(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        self.updating = False
        return result

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        q = self(batch, noise=self.noise_update).logits # (None, action_num)
        act = batch.act # (None,)
        q = q[np.arange(len(act)), act] # (None, )
        r = to_torch_as(batch.returns, q).squeeze(dim=-1) # (None, )
        if self.use_target_noise:
            if self.use_dueling:
                update_noise = torch.cat([self.noise_update['Q']['hyper_noise'], self.noise_update['V']['hyper_noise']], dim=1)
            else:
                update_noise = self.noise_update['Q']['hyper_noise']
            target_noise = to_torch_as(batch.target_noise, update_noise)
            loss_noise = torch.sum(target_noise.mul(update_noise).to(r.device), dim=1)
            r += loss_noise
        td = (r - q).pow(2) # (None,)
        weight = batch.pop("weight", 1.0)
        loss = (td * weight).mean(0).sum()
        if self.hyper_reg_coef:
            reg_loss = self.model.Q.regularization(self.noise_update['Q'])
            if self.use_dueling:
                reg_loss += self.model.V.regularization(self.noise_update['V'])
            loss += reg_loss * (self.hyper_reg_coef / kwargs['sample_num'])
        batch.weight = td  # prio-buffer
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}
