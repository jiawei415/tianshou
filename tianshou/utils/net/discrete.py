from typing import Any, Dict, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch
from tianshou.utils.net.common import MLP


class Actor(nn.Module):
    """Simple actor network.

    Will create an actor operated in discrete action space with structure of
    preprocess_net ---> action_shape.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param bool softmax_output: whether to apply a softmax layer over the last
        layer's output.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        softmax_output: bool = True,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = int(np.prod(action_shape))
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, self.output_dim, hidden_sizes, device=self.device)
        self.softmax_output = softmax_output

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state)
        logits = self.last(logits)
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, h


class Critic(nn.Module):
    """Simple critic network. Will create an actor operated in discrete \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int last_size: the output dimension of Critic network. Default to 1.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        hidden_sizes: Sequence[int] = (),
        last_size: int = 1,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.preprocess = preprocess_net
        self.output_dim = last_size
        input_dim = getattr(preprocess_net, "output_dim", preprocess_net_output_dim)
        self.last = MLP(input_dim, last_size, hidden_sizes, device=self.device)

    def forward(
        self, s: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.preprocess(s, state=kwargs.get("state", None))
        return self.last(logits)


class CosineEmbeddingNetwork(nn.Module):
    """Cosine embedding network for IQN. Convert a scalar in [0, 1] to a list \
    of n-dim vectors.

    :param num_cosines: the number of cosines used for the embedding.
    :param embedding_dim: the dimension of the embedding/output.

    .. note::

        From https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_cosines: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_cosines, embedding_dim), nn.ReLU())
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus: torch.Tensor) -> torch.Tensor:
        batch_size = taus.shape[0]
        N = taus.shape[1]
        # Calculate i * \pi (i=1,...,N).
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines + 1, dtype=taus.dtype, device=taus.device
        ).view(1, 1, self.num_cosines)
        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi
                            ).view(batch_size * N, self.num_cosines)
        # Calculate embeddings of taus.
        tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)
        return tau_embeddings


class ImplicitQuantileNetwork(Critic):
    """Implicit Quantile Network.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_dim: the dimension of action space.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        Although this class inherits Critic, it is actually a quantile Q-Network
        with output shape (batch_size, action_dim, sample_size).

        The second item of the first return value is tau vector.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu"
    ) -> None:
        last_size = np.prod(action_shape)
        super().__init__(
            preprocess_net, hidden_sizes, last_size, preprocess_net_output_dim, device
        )
        self.input_dim = getattr(
            preprocess_net, "output_dim", preprocess_net_output_dim
        )
        self.embed_model = CosineEmbeddingNetwork(num_cosines,
                                                  self.input_dim).to(device)

    def forward(  # type: ignore
        self, s: Union[np.ndarray, torch.Tensor], sample_size: int, **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state=kwargs.get("state", None))
        # Sample fractions.
        batch_size = logits.size(0)
        taus = torch.rand(
            batch_size, sample_size, dtype=logits.dtype, device=logits.device
        )
        embedding = (logits.unsqueeze(1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        out = self.last(embedding).view(batch_size, sample_size, -1).transpose(1, 2)
        return (out, taus), h


class FractionProposalNetwork(nn.Module):
    """Fraction proposal network for FQF.

    :param num_fractions: the number of factions to propose.
    :param embedding_dim: the dimension of the embedding/input.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(self, num_fractions: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(embedding_dim, num_fractions)
        torch.nn.init.xavier_uniform_(self.net.weight, gain=0.01)
        torch.nn.init.constant_(self.net.bias, 0)
        self.num_fractions = num_fractions
        self.embedding_dim = embedding_dim

    def forward(
        self, state_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Calculate (log of) probabilities q_i in the paper.
        m = torch.distributions.Categorical(logits=self.net(state_embeddings))
        taus_1_N = torch.cumsum(m.probs, dim=1)
        # Calculate \tau_i (i=0,...,N).
        taus = F.pad(taus_1_N, (1, 0))
        # Calculate \hat \tau_i (i=0,...,N-1).
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.0
        # Calculate entropies of value distributions.
        entropies = m.entropy()
        return taus, tau_hats, entropies


class FullQuantileFunction(ImplicitQuantileNetwork):
    """Full(y parameterized) Quantile Function.

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param int action_dim: the dimension of action space.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int num_cosines: the number of cosines to use for cosine embedding.
        Default to 64.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    .. note::

        The first return value is a tuple of (quantiles, fractions, quantiles_tau),
        where fractions is a Batch(taus, tau_hats, entropies).
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_sizes: Sequence[int] = (),
        num_cosines: int = 64,
        preprocess_net_output_dim: Optional[int] = None,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__(
            preprocess_net, action_shape, hidden_sizes, num_cosines,
            preprocess_net_output_dim, device
        )

    def _compute_quantiles(
        self, obs: torch.Tensor, taus: torch.Tensor
    ) -> torch.Tensor:
        batch_size, sample_size = taus.shape
        embedding = (obs.unsqueeze(1) *
                     self.embed_model(taus)).view(batch_size * sample_size, -1)
        quantiles = self.last(embedding).view(batch_size, sample_size,
                                              -1).transpose(1, 2)
        return quantiles

    def forward(  # type: ignore
        self, s: Union[np.ndarray, torch.Tensor],
        propose_model: FractionProposalNetwork,
        fractions: Optional[Batch] = None,
        **kwargs: Any
    ) -> Tuple[Any, torch.Tensor]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, h = self.preprocess(s, state=kwargs.get("state", None))
        # Propose fractions
        if fractions is None:
            taus, tau_hats, entropies = propose_model(logits.detach())
            fractions = Batch(taus=taus, tau_hats=tau_hats, entropies=entropies)
        else:
            taus, tau_hats = fractions.taus, fractions.tau_hats
        quantiles = self._compute_quantiles(logits, tau_hats)
        # Calculate quantiles_tau for computing fraction grad
        quantiles_tau = None
        if self.training:
            with torch.no_grad():
                quantiles_tau = self._compute_quantiles(logits, taus[:, 1:-1])
        return (quantiles, fractions, quantiles_tau), h


class NoisyLinear(nn.Module):
    """Implementation of Noisy Networks. arXiv:1706.10295.

    :param int in_features: the number of input features.
    :param int out_features: the number of output features.
    :param float noisy_std: initial standard deviation of noisy linear layers.

    .. note::

        Adapted from https://github.com/ku2482/fqf-iqn-qrdqn.pytorch/blob/master
        /fqf_iqn_qrdqn/network.py .
    """

    def __init__(
        self, in_features: int, out_features: int, noisy_std: float = 0.5
    ) -> None:
        super().__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std

        self.reset()
        self.sample()

    def reset(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

    def f(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.randn(x.size(0), device=x.device)
        return x.sign().mul_(x.abs().sqrt_())

    def sample(self) -> None:
        self.eps_p.copy_(self.f(self.eps_p))  # type: ignore
        self.eps_q.copy_(self.f(self.eps_q))  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.mu_W + self.sigma_W * (
                self.eps_q.ger(self.eps_p)  # type: ignore
            )
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()  # type: ignore
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, not np.all(self.mu_bias.cpu().detach().numpy() == 0)
        )


class PriorHyperLinear(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            prior_mean: float or np.ndarray = 0.,
            prior_std: float or np.ndarray = 1.,
    ):
        super().__init__()

        self.in_features, self.out_features = input_size, output_size
        # (fan-out, fan-in)
        self.weight = np.random.randn(output_size, input_size).astype(np.float32)
        self.weight = self.weight / np.linalg.norm(self.weight, axis=1, keepdims=True)

        if isinstance(prior_mean, np.ndarray):
            self.bias = prior_mean
        else:
            self.bias = np.ones(output_size, dtype=np.float32) * prior_mean

        if isinstance(prior_std, np.ndarray):
            if prior_std.ndim == 1:
                assert len(prior_std) == output_size
                prior_std = np.diag(prior_std).astype(np.float32)
            elif prior_std.ndim == 2:
                assert prior_std.shape == (output_size, output_size)
                prior_std = prior_std
            else:
                raise ValueError
        else:
            assert isinstance(prior_std, (float, int, np.float32, np.int32, np.float64, np.int64))
            prior_std = np.eye(output_size, dtype=np.float32) * prior_std

        self.weight = torch.nn.Parameter(torch.from_numpy(prior_std @ self.weight).float())
        self.bias = torch.nn.Parameter(torch.from_numpy(self.bias).float())

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight.to(x.device), self.bias.to(x.device))
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, not np.all(self.bias.cpu().detach().numpy() == 0)
        )


class PriorNoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        noisy_std: float,
        batch_noise: bool = False,
    ) -> None:
        super().__init__()

        weight_shape = (in_features, out_features) if batch_noise else (out_features, in_features)
        bias_shape = (1, out_features) if batch_noise else (out_features, )
        self.mu_W = nn.Parameter(torch.FloatTensor(size=weight_shape))
        self.sigma_W = nn.Parameter(torch.FloatTensor(size=weight_shape))
        self.mu_bias = nn.Parameter(torch.FloatTensor(size=bias_shape))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(size=bias_shape))

        self.base_forward = getattr(self, "base_forward_v1") if batch_noise else getattr(self, "base_forward_v2")

        # Factorized noise.
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = noisy_std

        self.init_params()

    def init_params(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

    def base_forward_v1(self, x, eps_p, eps_q):
        weight = self.mu_W + self.sigma_W * torch.bmm(eps_p.unsqueeze(dim=-1), eps_q.unsqueeze(dim=1))
        bias = self.mu_bias + self.sigma_bias * eps_q.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        out = torch.bmm(x, weight) + bias
        return out.squeeze()

    def base_forward_v2(self, x, eps_p, eps_q):
        weight = self.mu_W + self.sigma_W * torch.multiply(eps_q.T, eps_p)
        bias = self.mu_bias + self.sigma_bias * eps_q.squeeze()
        out =  F.linear(x, weight, bias)
        return out

    def forward(self, x: torch.Tensor, eps_p: torch.Tensor, eps_q: torch.Tensor) -> torch.Tensor:
        out = self.base_forward(x, eps_p, eps_q)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, not np.all(self.mu_bias.cpu().detach().numpy() == 0)
        )


class NewNoisyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        noisy_std: float,
        prior_std: float = 1.,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        batch_noise: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Learnable parameters.
        weight_shape = (in_features, out_features) if batch_noise else (out_features, in_features)
        bias_shape = (1, out_features) if batch_noise else (out_features, )
        self.mu_W = nn.Parameter(torch.FloatTensor(size=weight_shape))
        self.sigma_W = nn.Parameter(torch.FloatTensor(size=weight_shape))
        self.mu_bias = nn.Parameter(torch.FloatTensor(size=bias_shape))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(size=bias_shape))

        if prior_std:
            self.priormodel = PriorNoisyLinear(in_features, out_features, noisy_std=noisy_std, batch_noise=batch_noise)
            for param in self.priormodel.parameters():
                param.requires_grad = False

        self.base_forward = getattr(self, "base_forward_v1") if batch_noise else getattr(self, "base_forward_v2")

        # Factorized noise.
        self.eps_p = None
        self.eps_q = None
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.sigma = noisy_std
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

        self.init_params()

    # def reset_noise(self, noise):
    #     self.eps_p = noise['eps_p'].to(self.device)
    #     self.eps_q = noise['eps_q'].to(self.device)

    def init_params(self) -> None:
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.in_features))

    def base_forward_v1(self, x, eps_p, eps_q):
        weight = self.mu_W + self.sigma_W * torch.bmm(eps_p.unsqueeze(dim=-1), eps_q.unsqueeze(dim=1))
        bias = self.mu_bias + self.sigma_bias * eps_q.unsqueeze(dim=1)
        x = x.unsqueeze(dim=1)
        out = torch.bmm(x, weight) + bias
        return out.squeeze()

    def base_forward_v2(self, x, eps_p, eps_q):
        weight = self.mu_W + self.sigma_W * torch.multiply(eps_q.T, eps_p)
        bias = self.mu_bias + self.sigma_bias * eps_q.squeeze()
        out =  F.linear(x, weight, bias)
        return out

    def forward(self, x: torch.Tensor, prior_x=None, noise: Dict[str, Any] = {}) -> torch.Tensor:
        eps_q = noise["eps_q"].to(self.device)
        eps_p = noise["eps_p"].to(self.device)
        out = self.base_forward(x, eps_p.clone(), eps_q.clone())
        if prior_x is not None and self.prior_std > 0:
            prior_out = self.priormodel(prior_x, eps_p.clone(), eps_q.clone())
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, not np.all(self.mu_bias.cpu().detach().numpy() == 0)
        )


class NewHyperLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        noise_dim: int,
        prior_std: float = 1.,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        batch_noise: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # Learnable parameters.
        inp_dim = noise_dim
        out_dim = in_features * out_features + out_features
        self.hypermodel = nn.Linear(inp_dim, out_dim)
        if prior_std > 0:
            self.priormodel = PriorHyperLinear(inp_dim, out_dim, prior_std=prior_std)
            for param in self.priormodel.parameters():
                param.requires_grad = False

        self.base_forward = getattr(self, "base_forward_v1") if batch_noise else getattr(self, "base_forward_v2")

        self.hyper_noise = None
        self.device = device
        self.noise_dim = noise_dim
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.splited_size = [in_features * out_features, out_features]
        self.weight_shape = (in_features, out_features) if batch_noise else (out_features, in_features)
        self.bias_shape = (1, out_features) if batch_noise else (out_features,)

    # def reset_noise(self, noise):
    #     self.hyper_noise = noise['hyper_noise'].to(self.device)

    def base_forward_v1(self, x: torch.Tensor, params: torch.Tensor):
        weight, bias = params.split(self.splited_size, dim=1)
        weight = weight.reshape((-1,) + self.weight_shape)
        bias = bias.reshape((-1,) + self.bias_shape)
        x = x.unsqueeze(dim=1)
        out = torch.bmm(x, weight) + bias
        return out.squeeze()

    def base_forward_v2(self, x: torch.Tensor, params: torch.Tensor):
        weight, bias = params.split(self.splited_size, dim=1)
        weight = weight.reshape((-1,) + self.weight_shape).squeeze()
        bias = bias.reshape((-1,) + self.bias_shape).squeeze()
        out = F.linear(x, weight, bias)
        return out

    def forward(self, x: torch.Tensor, prior_x=None, noise: Dict[str, Any]={}) -> torch.Tensor:
        hyper_noise = noise['hyper_noise'].to(self.device)
        params = self.hypermodel(hyper_noise)
        out = self.base_forward(x, params)
        if prior_x is not None and self.prior_std > 0:
            prior_params = self.priormodel(hyper_noise)
            prior_out = self.base_forward(prior_x, prior_params)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out

    def regularization(self, noise: Dict[str, Any]={}) -> torch.Tensor:
        hyper_noise = noise['hyper_noise'].to(self.device)
        params = self.hypermodel(hyper_noise)
        reg_loss = params.pow(2).mean()
        return reg_loss


class EnsembleLinear(nn.Module):
    """
    Ensemble Network.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        ensemble_num: int,
        ensemble_sizes: Sequence[int] = (),
        prior_std: float = 0.0,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
    ):
        super().__init__()
        self.basedmodel = nn.ModuleList([
            self.mlp(in_features, out_features, ensemble_sizes) for _ in range(ensemble_num)
        ])
        if prior_std:
            self.priormodel = nn.ModuleList([
                self.mlp(in_features, out_features, ensemble_sizes) for _ in range(ensemble_num)
            ])
            for param in self.priormodel.parameters():
                param.requires_grad = False

        self.device = device
        self.ensemble_num = ensemble_num
        self.head_list = list(range(self.ensemble_num))
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def mlp(self, inp_dim, out_dim, hidden_sizes, bias=True):
        if len(hidden_sizes) == 0:
            return nn.Linear(inp_dim, out_dim, bias=bias)
        model = [nn.Linear(inp_dim, hidden_sizes[0], bias=bias)]
        model += [nn.ReLU(inplace=True)]
        for i in range(1, len(hidden_sizes)):
            model += [nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=bias)]
            model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(hidden_sizes[-1], out_dim, bias=bias)]
        return nn.Sequential(*model)

    def forward(self, x: torch.Tensor, prior_x=None, active_head=None) -> Tuple[torch.Tensor, Any]:
        if active_head is None or isinstance(active_head, list):
            active_head = active_head or self.head_list
            out = [self.basedmodel[k](x) for k in active_head]
            out = torch.stack(out, dim=1)
            if prior_x is not None and self.prior_std > 0:
                prior_out = [self.priormodel[k](x) for k in active_head]
                prior_out = torch.stack(prior_out, dim=1)
                out = out * self.posterior_scale + prior_out * self.prior_scale
        else:
            out = self.basedmodel[active_head](x)
            if prior_x is not None and self.prior_std > 0:
                prior_out = self.priormodel[active_head](x)
                out = out * self.posterior_scale + prior_out * self.prior_scale
        return out


class NewLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        prior_std: float = 0.0,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
    ):
        super().__init__()
        self.basedmodel = nn.Linear(in_features, out_features)
        if prior_std:
            self.priormodel = nn.Linear(in_features, out_features)
            for param in self.priormodel.parameters():
                param.requires_grad = False

        self.device = device
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale

    def forward(self, x: torch.Tensor, prior_x=None, ) -> Tuple[torch.Tensor, Any]:
        out = self.basedmodel(x)
        if prior_x is not None and self.prior_std > 0:
            prior_out = self.priormodel(prior_x)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out


class MultiHyperLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        noise_dim: int,
        prior_std: float = 1.,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        batch_noise: bool = True,
        action_num: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()

        # Learnable parameters.
        num_atoms = out_features // action_num
        inp_dim = noise_dim
        out_dim = in_features * num_atoms + num_atoms
        self.hypermodel = nn.ModuleList([
            nn.Linear(inp_dim, out_dim) for _ in range(action_num)
        ])
        if prior_std > 0:
            self.priormodel = nn.ModuleList([
                PriorHyperLinear(inp_dim, out_dim, prior_std=prior_std) for _ in range(action_num)
            ])
            for param in self.priormodel.parameters():
                param.requires_grad = False

        self.base_forward = getattr(self, "base_forward_v1") if batch_noise else getattr(self, "base_forward_v2")

        self.action_num = action_num
        self.device = device
        self.noise_dim = noise_dim
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.splited_size = [in_features * num_atoms, num_atoms]
        self.weight_shape = (in_features, num_atoms) if batch_noise else (num_atoms, num_atoms)
        self.bias_shape = (1, num_atoms) if batch_noise else (num_atoms,)

    def base_forward_v1(self, x: torch.Tensor, params: torch.Tensor):
        weight, bias = params.split(self.splited_size, dim=1)
        weight = weight.reshape((-1,) + self.weight_shape)
        bias = bias.reshape((-1,) + self.bias_shape)
        x = x.unsqueeze(dim=1)
        out = torch.bmm(x, weight) + bias
        return out.squeeze()

    def base_forward_v2(self, x: torch.Tensor, params: torch.Tensor):
        weight, bias = params.split(self.splited_size, dim=1)
        weight = weight.reshape((-1,) + self.weight_shape).squeeze()
        bias = bias.reshape((-1,) + self.bias_shape).squeeze()
        out = F.linear(x, weight, bias)
        return out

    def forward(self, x: torch.Tensor, prior_x=None, noise: Dict[str, Any]={}) -> torch.Tensor:
        hyper_noise = noise['hyper_noise'].to(self.device)
        out = []
        for i in range(self.action_num):
            params = self.hypermodel[i](hyper_noise)
            out.append(self.base_forward(x, params))
        out = torch.cat(out, dim=-1)
        if prior_x is not None and self.prior_std > 0:
            prior_out = []
            for i in range(self.action_num):
                prior_params = self.priormodel[i](hyper_noise)
                prior_out.append(self.base_forward(prior_x, prior_params))
            prior_out = torch.cat(prior_out, dim=-1)
            out = out * self.posterior_scale + prior_out * self.prior_scale
        return out

    def regularization(self, noise: Dict[str, Any]={}, p: int = 2) -> torch.Tensor:
        hyper_noise = noise['hyper_noise'].to(self.device)
        reg_loss = 0
        for i in range(self.action_num):
            params = self.hypermodel[0](hyper_noise)
            reg_loss += torch.norm(params, dim=1, p=p).square().mean()
        return reg_loss


class NewHyperLinearV2(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[str, int, torch.device]],
        noise_dim: int,
        prior_std: float = 1.,
        prior_scale: float = 1.,
        posterior_scale: float = 1.,
        batch_noise: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.hypermodel = HyperLinearLayer(noise_dim, (out_features, in_features),
                                           target_init="tf", device=device)
        if prior_std > 0:
            self.priormodel = HyperLinearLayer(noise_dim, (out_features, in_features),
                                           target_init="tf", device=device)
            for param in self.priormodel.parameters():
                param.requires_grad = False

        self.hyper_noise = None
        self.device = device
        self.noise_dim = noise_dim
        self.prior_std = prior_std
        self.prior_scale = prior_scale
        self.posterior_scale = posterior_scale
        self.splited_size = [in_features * out_features, out_features]
        self.weight_shape = (in_features, out_features) if batch_noise else (out_features, in_features)
        self.bias_shape = (1, out_features) if batch_noise else (out_features,)

    def forward(self, x: torch.Tensor, prior_x=None, noise: Dict[str, Any]={}) -> torch.Tensor:
        hyper_noise = noise['hyper_noise'].to(self.device)
        qs = []
        for k in range(hyper_noise.shape[0]):
            q = self.hypermodel(x, hyper_noise[k][None])
            qs.append(q)
        q = torch.stack(qs, dim=1)  # (None, num_ensemble, num_action)
        if prior_x is not None and self.prior_std > 0:
            prior_qs = []
            for k in range(hyper_noise.shape[0]):
                prior_q = self.priormodel(prior_x, hyper_noise[k][None])
                prior_qs.append(prior_q)
            prior_q = torch.stack(prior_qs, dim=1)  # (None, num_ensemble, num_action)
            q = q * self.posterior_scale + prior_q * self.prior_scale
        if hyper_noise.shape[0] == 1:
            q = q[:, 0, :]
        return q

    def regularization(self, noise: Dict[str, Any]={}) -> torch.Tensor:
        hyper_noise = noise['hyper_noise'].to(self.device)
        hyper_weight = self.hypermodel.hyper_weight(hyper_noise)
        hyper_bias = self.hypermodel.hyper_bias(hyper_noise)
        params = torch.cat([hyper_weight.reshape(len(hyper_noise), -1), hyper_bias.reshape(len(hyper_noise), -1)], dim=1)
        reg_loss = params.pow(2).mean()
        return reg_loss


class HyperLinearWeight(nn.Module):
    """
    A linear hypermodel to generate the ``weight`` for single layer of the base model.
    """
    def __init__(
        self,
        z_size: int,
        output_shape: Tuple[int],
        target_init: str = "tf",
        has_bias: bool = True,
        trainable: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.z_size = z_size
        self.output_shape = output_shape
        self.target_init = target_init
        self.has_bias = has_bias
        self.trainable = trainable
        self.device = device

        self.num_param = int(np.prod(self.output_shape))

        self.weight = nn.Parameter(torch.empty([self.num_param, self.z_size], dtype=torch.float32, device=self.device))
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros([self.num_param], dtype=torch.float32, device=self.device))
        else:
            self.bias = torch.zeros([self.num_param], dtype=torch.float32, device=self.device)

        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        z: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        param = F.linear(z, self.weight, self.bias)
        param = param.reshape([len(z), *self.output_shape])
        return param

    def reset_parameters(self):
        """
        Initialize the ``weight`` term.
        TensorFlow: 1/sqrt(f_in) initialization.
        PyTorch: He Kaiming initialization.
        """
        if self.target_init == "tf":
            f_in = self.output_shape[1]
            bound = 1.0 / math.sqrt(f_in)
            nn.init.trunc_normal_(self.weight, std=bound, a=-2 * bound, b=2 * bound)
        elif self.target_init == "torch":
            f_in = self.output_shape[1]
            gain = nn.init.calculate_gain('leaky_relu', math.sqrt(5))
            std = gain / math.sqrt(f_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                self.weight.uniform_(-bound, bound)
        else:
            raise ValueError(self.target_init)


class HyperLinearBias(nn.Module):
    """
    A linear hypermodel to generate the ``bias`` for single layer of the base model.
    """
    def __init__(
        self,
        z_size: int,
        output_shape: Tuple[int],
        target_init: str = "tf",
        has_bias: bool = True,
        trainable: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()
        self.z_size = z_size
        self.output_shape = output_shape
        self.output_dim = output_shape[0]
        self.target_init = target_init
        self.has_bias = has_bias
        self.trainable = trainable
        self.device = device

        self.num_param = self.output_dim

        self.weight = nn.Parameter(torch.empty([self.num_param, self.z_size], dtype=torch.float32, device=self.device))
        if self.has_bias:
            self.bias = nn.Parameter(torch.zeros([self.num_param], dtype=torch.float32, device=self.device))
        else:
            self.bias = torch.zeros([self.num_param], dtype=torch.float32, device=self.device)

        self.reset_parameters()

        if not self.trainable:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def forward(
        self,
        z: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        param = F.linear(z, self.weight, self.bias)
        return param

    def reset_parameters(self):
        """
        Initialize the ``bias`` term.
        TensorFlow: zero initialization.
        PyTorch: Unif(1/sqrt(f_in)) initialization.
        """
        if self.target_init == "tf":
            nn.init.zeros_(self.weight)
        elif self.target_init == "torch":
            f_in = self.output_shape[1]
            bound = 1.0 / math.sqrt(f_in)
            nn.init.uniform(self.weight, -bound, bound)
        else:
            raise ValueError(self.target_init)


class HyperLinearLayer(nn.Module):
    """
    A linear hypermodel to mimic a layer of the base model.
    """
    def __init__(
        self,
        z_size: int,
        output_shape: Union[Tuple[int], Any],
        target_init: str = "tf",
        has_bias: bool = True,
        trainable: bool = True,
        device: Union[str, int, torch.device] = "cpu",
    ):
        super().__init__()

        self.hyper_weight = HyperLinearWeight(z_size, output_shape, target_init, has_bias, trainable, device)
        self.hyper_bias = HyperLinearBias(z_size, output_shape, target_init, has_bias, trainable, device)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        weight = self.hyper_weight(z)[0]
        bias = self.hyper_bias(z)[0]

        h = F.linear(x, weight, bias)
        return h


def noisy_layer_noise(batch, inp_dim, out_dim):
    def f(x):
        return x.sign().mul_(x.abs().sqrt_())
    eps_p = f(torch.randn(size=(batch, inp_dim)))
    eps_q = f(torch.randn(size=(batch, out_dim)))
    return eps_p, eps_q


def hyper_layer_noise(batch, noise_dim, noise_std):
    noise = torch.randn(size=(batch, noise_dim)) * noise_std
    return noise


def sample_noise(model: nn.Module) -> bool:
    """Sample the random noises of NoisyLinear modules in the model.

    :param model: a PyTorch module which may have NoisyLinear submodules.
    :returns: True if model has at least one NoisyLinear submodule;
        otherwise, False.
    """
    done = False
    for m in model.modules():
        if isinstance(m, NoisyLinear):
            m.sample()
            done = True
    return done
