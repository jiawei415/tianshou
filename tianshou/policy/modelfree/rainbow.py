import torch
import numpy as np
from typing import Any, Dict

from tianshou.data import Batch
from tianshou.policy import C51Policy
from tianshou.utils.net.discrete import sample_noise


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


class HyperRainbowPolicy(C51Policy):
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
        estimation_step: int = 1,
        target_update_freq: int = 0,
        reward_normalization: bool = False,
        **kwargs: Any
    ) -> None:
        super(HyperRainbowPolicy, self).__init__(
            model, optim, discount_factor, num_atoms, v_min, v_max,
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
