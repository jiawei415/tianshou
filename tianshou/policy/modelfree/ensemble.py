import torch
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Optional, Union, Callable
from tianshou.data.utils.converter import to_torch

from tianshou.policy import BasePolicy, C51Policy, DQNPolicy
from tianshou.policy.base import _nstep_return
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy


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
    target_qs = []
    with torch.no_grad():
        target_q_torch = target_q_fn(buffer, terminal) # (None, ensemble_num) or (None, ensemble_num, num_atoms)
    end_flag = buffer.done.copy()
    end_flag[buffer.unfinished_index()] = True
    for k in range(target_q_torch.shape[1]):
        target_q = to_numpy(target_q_torch[:, k].reshape(bsz, -1))
        target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1)
        target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step)
        target_qs.append(target_q)
    target_qs = np.array(target_qs) # (ensemble_num, None, 1) or (ensemble_num, None, num_atoms)
    target_qs = target_qs.swapaxes(1, 0) # (None, ensemble_num, 1) or (None, ensemble_num, num_atoms)
    batch.returns = to_torch_as(target_qs, target_q_torch).squeeze(dim=-1)
    if hasattr(batch, "weight"):  # prio buffer update
        batch.weight = to_torch_as(batch.weight, target_q_torch)
    return batch


class EnsembleDQNPolicy(DQNPolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
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
        ensemble_num: int = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model, optim, discount_factor, estimation_step, target_update_freq, reward_normalization, **kwargs
        )
        self.use_dueling = use_dueling
        self.same_noise_update = same_noise_update
        self.batch_noise_update = batch_noise_update
        self.sample_per_step = sample_per_step
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.value_var_eps = value_var_eps
        self.value_gap_eps = value_gap_eps
        self.ensemble_num = ensemble_num
        self.active_head_train = None
        self.active_head_test = None
        self.active_head_update = None
        if self.action_select_scheme is None:
            assert self.action_sample_num == 1
            self.get_actions = getattr(self, '_greedy_action_select')
        elif self.action_select_scheme == 'MAX':
            self.get_actions = getattr(self, '_max_action_select')
        elif self.action_select_scheme == 'VIDS':
            self.get_actions = getattr(self, '_vids_action_select')
        else:
            raise NotImplementedError(f'No action selcet scheme {self.action_select_scheme}')

    def _vids_action_select(self, q):
        q = q.squeeze(0)
        value_gap = q.max(dim=-1, keepdim=True)[0] - q
        value_gap = value_gap.mean(dim=0) + self.value_gap_eps
        value_var = torch.var(q, dim=0) + self.value_var_eps
        act = to_numpy(torch.argmin(value_gap / value_var)).reshape(1)
        return act

    def _max_action_select(self, q):
        q = q.squeeze(0)
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
            main_model_head = self.active_head_update
            target_model_head = self.active_head_update
        else:
            main_model_head = self.sample_head(head_num=batch_size)
            target_model_head = self.sample_head(head_num=batch_size)
        with torch.no_grad():
            a = self(batch, input="obs_next", active_head=main_model_head).act # (None, ensemble_num)
            target_q = self(batch, model="model_old", input="obs_next", active_head=target_model_head).logits # (None, ensemble_num, action_num)
        a_one_hot = F.one_hot(torch.as_tensor(a, device=target_q.device), self.max_action_num).to(torch.float32) # (None, ensemble_num, action_num)
        target_q = torch.sum(target_q * a_one_hot, dim=-1)  # (None, ensemble_num)
        return target_q

    def forward(
        self, batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        active_head: Any = None,
        **kwargs: Any
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        if not self.updating:
            done = len(batch.done.shape) == 0 or batch.done[0]
            if self.training:
                if self.active_head_train is None or self.sample_per_step or done:
                    self.active_head_train = self.sample_head(head_num=self.action_sample_num)
                active_head = self.active_head_train
            else:
                if self.active_head_test is None or self.sample_per_step or done:
                    self.active_head_test = self.sample_head(head_num=self.action_sample_num)
                active_head = self.active_head_test
            logits, h = model(obs_, state=state, active_head=active_head, info=batch.info) # (None, action_sample_num, num_atoms)
            q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None,)
            act = self.get_actions(q)
        else:
            logits, h = model(obs_, state=state, active_head=active_head, info=batch.info) # (None, ensemble_num)
            q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None, ensemble_num)
            act = to_numpy(q.max(dim=-1)[1])
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        return Batch(logits=logits, act=act, state=h)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        self.updating = False
        return result

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        q = self(batch, active_head=self.active_head_update).logits # (None, ensemble_num, num_action)
        a_one_hot = F.one_hot(torch.as_tensor(batch.act, device=q.device), self.max_action_num).to(torch.float32) # (None, num_actions)
        q = torch.einsum('bka,ba->bk', q, a_one_hot) # (None, ensemble_num)
        r = to_torch_as(batch.returns, q) # (None, ensemble_num)
        td = (r - q).pow(2) # (None, ensemble_num)
        masks = to_torch_as(batch.ensemble_mask, q) # (None, ensemble_num)
        td *= masks # (None, ensemble_num)
        weight = batch.pop("weight", 1.0)
        loss = (td * weight).mean(0).sum()
        batch.weight = td  # prio-buffer
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def sample_head(self, head_num: Any =None):
        head_num = head_num or self.ensemble_num
        return np.random.randint(low=0, high=self.ensemble_num, size=head_num).tolist()

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
        return compute_nstep_return(batch, buffer, indice, target_q_fn, gamma, n_step, rew_norm)


class EnsembleC51Policy(C51Policy):
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
        ensemble_num: int = 1,
        **kwargs: Any
    ) -> None:
        super().__init__(
            model, optim, discount_factor, num_atoms, v_min, v_max,
            estimation_step, target_update_freq, reward_normalization, **kwargs
        )
        self.use_dueling = use_dueling
        self.same_noise_update = same_noise_update
        self.batch_noise_update = batch_noise_update
        self.sample_per_step = sample_per_step
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.value_var_eps = value_var_eps
        self.value_gap_eps = value_gap_eps
        self.ensemble_num = ensemble_num
        self.active_head_train = None
        self.active_head_test = None
        self.active_head_update = None
        if self.action_select_scheme is None:
            assert self.action_sample_num == 1
            self.get_actions = getattr(self, '_greedy_action_select')
        elif self.action_select_scheme == 'MAX':
            self.get_actions = getattr(self, '_max_action_select')
        elif self.action_select_scheme == 'VIDS':
            self.get_actions = getattr(self, '_vids_action_select')
        else:
            raise NotImplementedError(f'No action selcet scheme {self.action_select_scheme}')

    def _vids_action_select(self, logits, q):
        logits = logits.squeeze(0)
        q = q.squeeze(0)
        value_gap = q.max(dim=-1, keepdim=True)[0] - q
        value_gap = value_gap.mean(dim=0) + self.value_gap_eps
        value_var = torch.var(logits, dim=0) # (action_sum, num_atoms)
        value_var = value_var.sum(dim=1) + self.value_var_eps # (action_sum,)
        act = to_numpy(torch.argmin(value_gap / value_var)).reshape(1)
        return act

    def _max_action_select(self, logits, q):
        q = q.squeeze(0)
        action_num = q.shape[-1]
        act = to_numpy(torch.argmax(q) % action_num).reshape(1)
        return act

    def _greedy_action_select(self, logits, q):
        act = to_numpy(q.max(dim=-1)[1])
        return act

    def _target_dist(self, batch: Batch) -> torch.Tensor:
        if self.same_noise_update:
            main_model_head = self.active_head_update
            target_model_head = self.active_head_update
        else:
            main_model_head = self.sample_head()
            target_model_head = self.sample_head()
        with torch.no_grad():
            a = self(batch, input="obs_next", active_head=main_model_head).act # (None, ensemble_num)
            next_dist = self(batch, model="model_old", input="obs_next", active_head=target_model_head).logits # (None, ensemble_num, action_num, num_atoms)
        a_one_hot = F.one_hot(torch.as_tensor(a, device=next_dist.device), self.max_action_num).to(torch.float32) # (None, ensemble_num, action_num)
        next_dist = torch.einsum('bkat,bka->bkt', next_dist, a_one_hot) # (None, ensemble_num, num_atoms)
        support = self.support.view(1, 1, -1, 1).repeat(1, self.ensemble_num, 1, 1) # (1, ensemble_num, num_atoms, 1)
        target_support = batch.returns.clamp(self._v_min, self._v_max).unsqueeze(-2) # (None, ensemble_num, 1, num_atoms)
        target_dist = (1 - (target_support - support).abs() / self.delta_z).clamp(0, 1) * next_dist.unsqueeze(-2) # (None, ensemble_num, num_atoms, num_atoms)
        return target_dist.sum(-1) # (None, ensemble_num, num_atoms)

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        return self.support.repeat(len(indices), self.ensemble_num, 1)

    def forward(
        self, batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        active_head: Any = None,
        **kwargs: Any
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        if not self.updating:
            done = len(batch.done.shape) == 0 or batch.done[0]
            if self.training:
                if self.active_head_train is None or self.sample_per_step or done:
                    self.active_head_train = self.sample_head(head_num=self.action_sample_num)
                active_head = self.active_head_train
            else:
                if self.active_head_test is None or self.sample_per_step or done:
                    self.active_head_test = self.sample_head(head_num=self.action_sample_num)
                active_head = self.active_head_test
            logits, h = model(obs_, state=state, active_head=active_head, info=batch.info)
            q = self.compute_q_value(logits, getattr(obs, "mask", None))
            act = self.get_actions(logits, q)
        else:
            logits, h = model(obs_, state=state, active_head=active_head, info=batch.info) # (None, ensemble_num, num_atoms)
            q = self.compute_q_value(logits, getattr(obs, "mask", None)) # (None, ensemble_num)
            act = to_numpy(q.max(dim=-1)[1])
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        return Batch(logits=logits, act=act, state=h)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer], **kwargs: Any) -> Dict[str, Any]:
        batch, indices = buffer.sample(sample_size)
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
            target_dist = self._target_dist(batch) # (None, ensemble_num, num_atoms)
        curr_dist = self(batch).logits # (None, ensemble_num, action_num, num_atoms)
        act_one_hot = F.one_hot(torch.as_tensor(batch.act, device=curr_dist.device), self.max_action_num).to(torch.float32) # (None, ensemble_num, action_num)
        curr_dist = torch.einsum('bkat,ba->bkt', curr_dist, act_one_hot) # (None, ensemble_num, num_atoms)
        masks = to_torch_as(batch.ensemble_mask, curr_dist) # (None, ensemble_num)
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(-1) # (None, ensemble_num)
        cross_entropy *= masks # (None, ensemble_num)
        weight = batch.pop("weight", 1.0)
        loss = (cross_entropy * weight).mean(0).sum()
        batch.weight = cross_entropy.detach()  # prio-buffer
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def sample_head(self, head_num: Any =None):
        head_num = head_num or self.ensemble_num
        return np.random.randint(low=0, high=self.ensemble_num, size=head_num).tolist()

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        logits = (logits * self.support).sum(-1)
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

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
        return compute_nstep_return(batch, buffer, indice, target_q_fn, gamma,n_step, rew_norm)