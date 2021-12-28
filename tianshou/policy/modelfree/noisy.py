import torch
import numpy as np
from typing import Any, Dict, Optional, Union, Callable

from tianshou.policy import C51Policy, DQNPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

def noisy_layer_noise(batch_size, inp_dim, out_dim):
    def f(x):
        return x.sign().mul_(x.abs().sqrt_())
    eps_p = f(torch.randn(size=(batch_size, inp_dim)))
    eps_q = f(torch.randn(size=(batch_size, out_dim)))
    return eps_p, eps_q


class NoisyDQNPolicy(DQNPolicy):
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
        **kwargs: Any
    ) -> None:
        super().__init__(
            model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, is_double, **kwargs
        )
        assert model.q_input_dim == model.v_input_dim
        self.last_layer_inp_dim = model.q_input_dim
        self.v_out_dim = model.v_output_dim if use_dueling else 0
        self.q_out_dim = model.q_output_dim
        self.use_dueling = use_dueling
        self.same_noise_update = same_noise_update
        self.batch_noise_update = batch_noise_update
        self.sample_per_step = sample_per_step
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.value_var_eps = value_var_eps
        self.value_gap_eps = value_gap_eps
        self.noise_test = None
        self.noise_train = None
        self.noise_update = None
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
        r = to_torch_as(batch.returns.flatten(), q) # (None, )
        td = (r - q).pow(2) # (None,)
        weight = batch.pop("weight", 1.0)
        loss = (td * weight).mean(0).sum()
        batch.weight = td  # prio-buffer
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def sample_noise(self, batch_size: int):
        Q_eps_p, Q_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.q_out_dim)
        V_eps_p, V_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.v_out_dim)
        noise = {'Q': {'eps_p': Q_eps_p, 'eps_q': Q_eps_q}, 'V': {'eps_p': V_eps_p, 'eps_q': V_eps_q}}
        return noise


class NoisyC51Policy(C51Policy):
    def __init__(
        self, model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        num_atoms: int = 51,
        v_min: float = -10,
        v_max: float = 10,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        use_dueling: bool = True,
        same_noise_update: bool = True,
        batch_noise_update: bool = True,
        sample_per_step: bool = False,
        action_sample_num: int = 1,
        action_select_scheme: str = None,
        value_var_eps: float = 1e-3,
        value_gap_eps: float = 1e-3,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model, optim, discount_factor, num_atoms, v_min, v_max,
            estimation_step, target_update_freq, reward_normalization, **kwargs
        )
        assert model.q_input_dim == model.v_input_dim
        self.last_layer_inp_dim = model.q_input_dim
        self.v_out_dim = model.v_output_dim if use_dueling else 0
        self.q_out_dim = model.q_output_dim
        self.use_dueling = use_dueling
        self.same_noise_update = same_noise_update
        self.batch_noise_update = batch_noise_update
        self.sample_per_step = sample_per_step
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.value_var_eps = value_var_eps
        self.value_gap_eps = value_gap_eps
        self.noise_test = None
        self.noise_train = None
        self.noise_update = None
        if self.action_select_scheme == "Greedy":
            assert self.action_sample_num == 1
            self.get_actions = getattr(self, '_greedy_action_select')
        elif self.action_select_scheme == 'MAX':
            self.get_actions = getattr(self, '_max_action_select')
        elif self.action_select_scheme == 'VIDS':
            self.get_actions = getattr(self, '_vids_action_select')
        else:
            raise NotImplementedError(f'No action selcet scheme {self.action_select_scheme}')

    def _vids_action_select(self, logits, q):
        value_gap = q.max(dim=-1, keepdim=True)[0] - q
        value_gap = value_gap.mean(dim=0) + self.value_gap_eps
        value_var = torch.var(logits, dim=0) # (action_sum, num_atoms)
        value_var = value_var.sum(dim=1) + self.value_var_eps # (action_sum,)
        act = to_numpy(torch.argmin(value_gap / value_var)).reshape(1)
        return act

    def _max_action_select(self, logits, q):
        action_num = q.shape[-1]
        act = to_numpy(torch.argmax(q) % action_num).reshape(1)
        return act

    def _greedy_action_select(self, logits, q):
        act = to_numpy(q.max(dim=-1)[1])
        return act

    def _target_dist(self, batch: Batch) -> torch.Tensor:
        batch_size = batch.obs.shape[0]
        if self.same_noise_update:
            main_model_noise = self.noise_update
            target_model_noise = self.noise_update
        else:
            main_model_noise = self.sample_noise(batch_size)
            target_model_noise = self.sample_noise(batch_size)
        if self._is_double:
            with torch.no_grad():
                a = self(batch, model="model", input="obs_next", noise=main_model_noise).act # (None,)
                next_dist = self(batch, model="model_old", input="obs_next", noise=target_model_noise).logits # (None, action_num, num_atoms)
        else:
            with torch.no_grad():
                next_b = self(batch, model="model_old", input="obs_next", noise=target_model_noise)
                a = next_b.act # (None, action_num)
                next_dist = next_b.logits # (None, action_num, num_atoms)
        next_dist = next_dist[np.arange(len(a)), a, :] # (None, num_atoms)
        support = self.support.view(1, -1, 1) # (1, num_atoms, 1)
        target_support = batch.returns.clamp(self._v_min, self._v_max).unsqueeze(-2) # (None, 1, num_atoms)
        target_dist = (1 - (target_support - support).abs() / self.delta_z).clamp(0, 1) * next_dist.unsqueeze(-2) # (None, num_atoms, num_atoms)
        return target_dist.sum(-1) # (None, num_atoms)

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
            act = self.get_actions(logits, q)
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
        with torch.no_grad():
            target_dist = self._target_dist(batch) # (None, num_atoms)
        curr_dist = self(batch, noise=self.noise_update).logits # (None, action_num, num_atoms)
        act = batch.act # (None,)
        curr_dist = curr_dist[np.arange(len(act)), act, :] # (None, num_atoms)
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(-1) # (None,)
        weight = batch.pop("weight", 1.0)
        loss = (cross_entropy * weight).mean(0).sum()
        batch.weight = cross_entropy.detach()  # prio-buffer
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def sample_noise(self, batch_size: int):
        Q_eps_p, Q_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.q_out_dim)
        V_eps_p, V_eps_q = noisy_layer_noise(batch_size, self.last_layer_inp_dim, self.v_out_dim)
        noise = {'Q': {'eps_p': Q_eps_p, 'eps_q': Q_eps_q}, 'V': {'eps_p': V_eps_p, 'eps_q': V_eps_q}}
        return noise
