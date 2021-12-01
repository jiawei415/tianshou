import torch
import numpy as np
from typing import Any, Dict, Optional, Union


from tianshou.data import Batch, to_numpy
from tianshou.policy import C51Policy, HyperC51Policy
from tianshou.utils.net.discrete import sample_noise, noisy_layer_noise, hyper_layer_noise


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
        noise_dim: int = 2,
        noise_std: float = 1.,
        hyper_reg_coef: float = 0.001,
        estimation_step: int = 1,
        target_update_freq: int = 0,
        action_select_scheme: str = "step",
        reward_normalization: bool = False,
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
        self.action_select_scheme = action_select_scheme

    def _target_dist(self, batch: Batch) -> torch.Tensor:
        if self._target:
            a = self(batch, input="obs_next", is_collecting=False).act
            next_dist = self(batch, model="model_old", input="obs_next", is_collecting=False).logits
        else:
            next_b = self(batch, input="obs_next", is_collecting=False)
            a = next_b.act
            next_dist = next_b.logits
        next_dist = next_dist[np.arange(len(a)), a, :]
        target_support = batch.returns.clamp(self._v_min, self._v_max)
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (
            1 - (target_support.unsqueeze(1) - self.support.view(1, -1, 1)).abs() /
            self.delta_z
        ).clamp(0, 1) * next_dist.unsqueeze(1)
        return target_dist.sum(-1)

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        model: str = "model",
        input: str = "obs",
        is_collecting: bool = True,
        **kwargs: Any
    ) -> Batch:
        done = batch['done'][0] if len(batch['done'].shape) > 0 else True
        obs = batch[input]
        obs_ = obs.obs if hasattr(obs, "obs") else obs
        if is_collecting and self.action_select_scheme == "step":
            self.reset_noise(obs_.shape[0], reset_target=False)
        elif is_collecting and done:
            self.reset_noise(obs_.shape[0], reset_target=False)
        model = getattr(self, model)
        logits, h = model(obs_, state=state, info=batch.info)
        q = self.compute_q_value(logits, getattr(obs, "mask", None))
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q.shape[1]
        act = to_numpy(q.max(dim=1)[1])
        return Batch(logits=logits, act=act, state=h)

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        self.reset_noise(batch['obs'].shape[0], reset_target=True)
        if self._target and self._iter % self._freq == 0:
            self.sync_weight()
        self.optim.zero_grad()
        with torch.no_grad():
            target_dist = self._target_dist(batch)
        weight = batch.pop("weight", 1.0)
        curr_dist = self(batch, is_collecting=False).logits
        act = batch.act
        curr_dist = curr_dist[np.arange(len(act)), act, :]
        cross_entropy = -(target_dist * torch.log(curr_dist + 1e-8)).sum(1)
        loss = (cross_entropy * weight).mean()
        if self.hyper_reg_coef and self.noise_dim:
            reg_loss = self.model.Q.model.regularization() + self.model.V.model.regularization()
            loss += reg_loss * (self.hyper_reg_coef / kwargs['sample_num'])
        batch.weight = cross_entropy.detach()  # prio-buffer
        loss.backward()
        self.optim.step()
        self._iter += 1
        return {"loss": loss.item()}

    def reset_noise(self, batch_size: int = 1, reset_target: bool = True):
        if self.noise_dim:
            noise = hyper_layer_noise(batch_size, self.noise_dim * 2, self.noise_std)
            Q_noise, V_noise = noise.split([self.noise_dim, self.noise_dim], dim=1)
            self.model.Q.model.reset_noise(Q_noise)
            self.model.V.model.reset_noise(V_noise)
            if reset_target:
                self.model_old.Q.model.reset_noise(Q_noise)
                self.model_old.V.model.reset_noise(V_noise)
        else:
            Q_eps_p, Q_eps_q = noisy_layer_noise(self.last_layer_inp_dim, self.q_out_dim)
            V_eps_p, V_eps_q = noisy_layer_noise(self.last_layer_inp_dim, self.v_out_dim)
            self.model.Q.model.reset_noise(Q_eps_p, Q_eps_q)
            self.model.V.model.reset_noise(V_eps_p, V_eps_q)
            if reset_target:
                self.model_old.Q.model.reset_noise(Q_eps_p, Q_eps_q)
                self.model_old.V.model.reset_noise(V_eps_p, V_eps_q)

