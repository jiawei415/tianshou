from copy import deepcopy
from typing import Any, Dict, Optional, Union, Callable

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.policy.base import _nstep_return
from tianshou.utils.net.discrete import sample_noise, noisy_layer_noise, hyper_layer_noise


class DQNPolicy(BasePolicy):
    """Implementation of Deep Q Network. arXiv:1312.5602.

    Implementation of Double Q-Learning. arXiv:1509.06461.

    Implementation of Dueling DQN. arXiv:1511.06581 (the dueling DQN is
    implemented in the network side, not here).

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float discount_factor: in [0, 1].
    :param int estimation_step: the number of steps to look ahead. Default to 1.
    :param int target_update_freq: the target network update frequency (0 if
        you do not use the target network). Default to 0.
    :param bool reward_normalization: normalize the reward to Normal(0, 1).
        Default to False.
    :param bool is_double: use double dqn. Default to True.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        result = self(batch, input="obs_next")
        if self._target:
            # target_Q = Q_old(s_, argmax(Q_new(s_, *)))
            target_q = self(batch, model="model_old", input="obs_next").logits
        else:
            target_q = result.logits
        if self._is_double:
            return target_q[np.arange(len(result.act)), result.act]
        else:  # Nature DQN, over estimate
            return target_q.max(dim=1)[0]

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        If you need to mask the action, please add a "mask" into batch.obs, for
        example, if we have an environment that has "0/1/2" three actions:
        ::

            batch == Batch(
                obs=Batch(
                    obs="original obs, with batch_size=1 for demonstration",
                    mask=np.array([[False, True, False]]),
                    # action 1 is available
                    # action 0 and 2 are unavailable
                ),
                ...
            )

        :param float eps: in [0, 1], for epsilon-greedy exploration method.

        :return: A :class:`~tianshou.data.Batch` which has 3 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        logits, h = model(obs_, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        weight = batch.pop("weight", 1.0)
        q = self(batch).logits
        q = q[np.arange(len(q)), batch.act]
        r = to_torch_as(batch.returns.flatten(), q)
        td = r - q
        loss = (td.pow(2) * weight).mean()
        batch.weight = td  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act


class NewDQNPolicy(BasePolicy):
    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        discount_factor: float = 0.99,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        is_double: bool = True,
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
        batch_noise_update: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.optim = optim
        self.eps = 0.0
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        assert estimation_step > 0, "estimation_step should be greater than 0"
        self._n_step = estimation_step
        self._target = target_update_freq > 0
        self._freq = target_update_freq
        self._iter = 0
        if self._target:
            self.model_old = deepcopy(self.model)
            self.model_old.eval()
        self._rew_norm = reward_normalization
        self._is_double = is_double

        self.last_layer_inp_dim = model.basedmodel.output_dim
        self.v_out_dim = model.V.output_dim
        self.q_out_dim = model.Q.output_dim
        self.noise_dim = noise_dim
        self.noise_std = noise_std
        self.target_noise_std = target_noise_std
        self.hyper_reg_coef = hyper_reg_coef
        self.sample_per_step = sample_per_step
        self.same_noise_update = same_noise_update
        self.batch_noise_update = batch_noise_update
        self.noise_train = None
        self.noise_test = None
        self.noise_update = None
        self.action_sample_num = action_sample_num
        self.action_select_scheme = action_select_scheme
        self.ensemble_num = ensemble_num
        self.active_head_train = None
        self.active_head_test = None
        self.active_head_update = None
        self.value_gap_eps = value_gap_eps
        self.value_var_eps = value_var_eps

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def train(self, mode: bool = True) -> "DQNPolicy":
        """Set the module in training mode, except for the target network."""
        self.training = mode
        self.model.train(mode)
        return self

    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.model_old.load_state_dict(self.model.state_dict())

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        batch = buffer[indices]  # batch.obs_next: s_{t+n}
        if self.ensemble_num:
            main_model_head = self.active_head_update if self.same_noise_update else np.random.randint(low=0, high=self.ensemble_num, size=self.ensemble_num).tolist()
            target_model_head = self.active_head_update if self.same_noise_update else np.random.randint(low=0, high=self.ensemble_num, size=self.ensemble_num).tolist()
            a = self(batch, input="obs_next", active_head=main_model_head).act # (None, ensemble_num)
            target_q = self(batch, model="model_old", input="obs_next", active_head=target_model_head).logits # (None, ensemble_num, action_num)
            a_one_hot = F.one_hot(torch.as_tensor(a, device=target_q.device), self.max_action_num).to(torch.float32) # (None, ensemble_num, action_num)
            target_q = torch.sum(target_q * a_one_hot, dim=-1) # (None, ensemble_num, action_num)
        else:
            main_model_noise  = self.noise_update if self.same_noise_update else self.sample_noise(batch['obs'].shape[0])
            target_model_noise = self.noise_update if self.same_noise_update else self.sample_noise(batch['obs'].shape[0])
            a = self(batch, input="obs_next", noise=main_model_noise).act # (None,)
            target_q = self(batch, model="model_old", input="obs_next", noise=target_model_noise).logits # (None, action_num)
            target_q = target_q[np.arange(len(a)), a]
        return target_q

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Compute the n-step return for Q-learning targets.

        More details can be found at
        :meth:`~tianshou.policy.BasePolicy.compute_nstep_return`.
        """
        batch = self.compute_nstep_return(
            batch, buffer, indices, self._target_q, self._gamma, self._n_step,
            self._rew_norm
        )
        return batch

    def compute_q_value(
        self, logits: torch.Tensor, mask: Optional[np.ndarray]
    ) -> torch.Tensor:
        """Compute the q value based on the network's raw output and action mask."""
        if mask is not None:
            # the masked q value should be smaller than logits.min()
            min_value = logits.min() - logits.max() - 1.0
            logits = logits + to_torch_as(1 - mask, logits) * min_value
        return logits

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        active_head: Any = None,
        noise: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> Batch:
        model = getattr(self, model)
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        done = batch['done'][0] if len(batch['done'].shape) > 0 else True
        if self.ensemble_num:
            if not self.updating:
                if not self.action_sample_num:
                    if self.training:
                        if self.active_head_train is None or self.sample_per_step or done:
                            self.active_head_train = np.random.randint(low=0, high=self.ensemble_num)
                        active_head = self.active_head_train
                    else:
                        if self.active_head_test is None or self.sample_per_step or done:
                            self.active_head_test = np.random.randint(low=0, high=self.ensemble_num)
                        active_head = self.active_head_test
            logits, h = model(obs_, state=state, active_head=active_head, info=batch.info) # (None, ensemble_num)
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
            logits, h = model(obs_, state=state, noise=noise, info=batch.info) # (None)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))  # (None,) or (None, ensemble_num)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[-1]
        if self.action_sample_num and not self.updating:
            q_ = q.squeeze(0)
            logits_ = logits.squeeze(0)
            if self.action_select_scheme == "MAX":
                act = to_numpy(torch.argmax(q_) % self.max_action_num).reshape(1)
            elif self.action_select_scheme == "VIDS":
                value_gap = q_.max(dim=-1, keepdim=True)[0] - q_
                value_gap = value_gap.mean(dim=0) + self.value_gap_eps
                value_var = torch.var(logits_, dim=0) + self.value_var_eps
                act = to_numpy(torch.argmin(value_gap / value_var)).reshape(1)
            else:
                raise ValueError(self.action_select_scheme)
        else:
            act = to_numpy(q.max(dim=-1)[1])
        return Batch(logits=logits, act=act, state=h)

    def update(self, sample_size: int, buffer: Optional[ReplayBuffer],
               **kwargs: Any) -> Dict[str, Any]:
        kwargs.update({"sample_num": len(buffer)})
        if buffer is None:
            return {}
        batch, indices = buffer.sample(sample_size)
        self.updating = True
        if not self.ensemble_num:
            noise_num = batch['obs'].shape[0] if self.batch_noise_update else 1
            self.noise_update = self.sample_noise(noise_num)
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, **kwargs)
        self.post_process_fn(batch, buffer, indices)
        self.updating = False
        return result

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        if self.ensemble_num:
            q = self(batch, active_head=self.active_head_update).logits # (None, ensemble_num, num_action)
            act = batch.act # (None,)
            a_one_hot = F.one_hot(torch.as_tensor(act, device=q.device), self.max_action_num).to(torch.float32) # (None, num_actions)
            q = torch.einsum('bka,ba->bk', q, a_one_hot) # (None, ensemble_num)
            r = to_torch_as(batch.returns, q) # (None, ensemble_num)
            td = (r - q).pow(2)
            masks = to_torch_as(batch.ensemble_mask, q)
            td *= masks
        else:
            q = self(batch, noise=self.noise_update).logits # (None, action_num)
            act = batch.act # (None,)
            q = q[np.arange(len(act)), act] # (None, )
            r = to_torch_as(batch.returns.flatten(), q) # (None, )
            if self.target_noise_std and self.noise_dim:
                update_noise = torch.cat([self.noise_update['Q']['hyper_noise'], self.noise_update['V']['hyper_noise']], dim=1)
                target_noise = to_torch_as(batch.target_noise, update_noise)
                loss_noise = torch.sum(target_noise.mul(update_noise).to(r.device), dim=1)
                r += loss_noise
            td = (r - q).pow(2) # (None,)
        weight = batch.pop("weight", 1.0)
        loss = (td * weight).mean(0).sum()
        if self.hyper_reg_coef and self.noise_dim:
            reg_loss = self.model.Q.model.regularization(self.noise_update['Q']) + self.model.V.model.regularization(self.noise_update['V'])
            loss += reg_loss * (self.hyper_reg_coef / kwargs['sample_num'])
        batch.weight = td  # prio-buffer
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
            target_q_torch = target_q_fn(buffer, terminal) # (None, ) or (None, ensemble_num)
        end_flag = buffer.done.copy()
        end_flag[buffer.unfinished_index()] = True
        if len(target_q_torch.shape) > 1:
            # Ziniu Li: return contains all target_q for ensembles
            target_qs = []
            for k in range(target_q_torch.shape[1]):
                target_q = to_numpy(target_q_torch[:, k].reshape(bsz, -1)) # (None, )
                target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1) # (None,)
                target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step) # (None,)
                target_qs.append(target_q.flatten())
            target_qs = np.array(target_qs) # (ensemble_num, None)
            target_qs = target_qs.transpose([1, 0]) # (None, ensemble_num)
            batch.returns = to_torch_as(target_qs, target_q_torch)
        else:
            target_q = to_numpy(target_q_torch.reshape(bsz, -1)) # (None,)
            target_q = target_q * BasePolicy.value_mask(buffer, terminal).reshape(-1, 1) # (None,)
            target_q = _nstep_return(rew, end_flag, target_q, indices, gamma, n_step) # (None,)
            batch.returns = to_torch_as(target_q, target_q_torch)
        if hasattr(batch, "weight"):  # prio buffer update
            batch.weight = to_torch_as(batch.weight, target_q_torch)
        return batch

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
