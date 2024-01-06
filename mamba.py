# <center><h1> The Annotated Mamba </h1></center>
#
#
# <center>
# <p><a href="https://arxiv.org/abs/2312.00752">Mamba: Linear-Time Sequence Modeling with Selective State Spaces</a></p>
# </center>
#
# <center>
# <p>Albert Gu and Tri Dao.</p>
# </center>
# <img src="mamba.png" width="100%"/>
# *Blog Post and [Library](https://github.com/srush/annotated-mamba/) by [Sasha Rush](http://rush-nlp.com/)
# ## Table of Contents
# * [Part 1: Time-Varying State Space Models] (Modeling)
#     - [Discrete-time SSM: The Recurrent Representation]
#     - [Tangent: A Mechanics Example]
#     - [Training SSMs: The Convolutional Representation]
#     - [An SSM Neural Network.]
# * [Part 1b: Addressing Long-Range Dependencies with HiPPO]
# * [Part 2: Implementing S4] (Advanced)
#     - [Step 1. SSM Generating Functions]
#     - [Step 2: Diagonal Case]
#     - [Step 3: Diagonal Plus Low-Rank]
#     - [Diagonal Plus Low-Rank RNN.]
#     - [Turning HiPPO to DPLR]
#     - [Final Check]
# * [Part 3: S4 in Practice] (NN Implementation)
#     - [S4 CNN / RNN Layer]
#     - [Sampling and Caching]
#     - [Experiments: MNIST]
#     - [Experiments: QuickDraw]
#     - [Experiments: Spoken Digits]
# * [Conclusion]
# <nav id="TOC">
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import arange
from torch import random
from torch import Tensor as T

# ## Part 1: State Space Models


# > The [state space model](https://en.wikipedia.org/wiki/State-space_representation) is defined by this simple equation.
# > It maps a 1-D input signal $u(t)$ to an $N$-D latent state $x(t)$
# > before projecting to a 1-D output signal $y(t)$.
# $$
#   \begin{aligned}
#     h'(t) &= \boldsymbol{A}h(t) + \boldsymbol{B}x(t) \\
#     y(t) &= \boldsymbol{C}h(t)
#   \end{aligned}
# $$

A_ = Float[T, '#B #L #D N 1']
B_ = Float[T, '#B #L #D N 1']
C_ = Float[T, '#B #L #D 1 N']
SSM_ = tuple[A_, B_, C_]


# Shape check.

def SSM(A: A_, B: B_, C: C_) -> SSM_:
    return (A, B, C)

# Shape


def random_ssm(N: int, L: int = 1) -> SSM_:
    A, B, C = random(1, 1, 1, N), random(1, 1, 1, N, 1), random(1, 1, 1, 1, N)
    return SSM(A, B, C)


# Same type

# $$
#   \begin{aligned}
#     h_t &= \boldsymbol{\overline{A}}h_t + \boldsymbol{\overline{B}}x_t \\
#     y_t &= \boldsymbol{C}h_t
#   \end{aligned}
# $$

X_ = Float[T, 'B L D']
Y_ = Float[T, 'B L D']
H_ = Float[T, 'B D N 1']


def ssm_rnn(A_bar: A_, B_bar: B_, C: C_, x: X_) -> Y_:
    B, L, D = x.shape
    N = A_bar.shape[-1]
    ys = []
    h_l_1: H_ = torch.zeros(B, D, N, 1)
    for l, x_l in enumerate(x.unbind(-1)):
        h_l: H_ = A_bar[:, l] * h_l_1 + B_bar[:, l] @ x_l[..., None, None]
        y_l: Float[T, 'B D'] = C[:, l] @ h_l
        ys.append(y_l[..., 0, 0])
        h_l_1 = h_l
    return torch.stack(ys)
#
# $$
#   \begin{aligned}
#     L &= \text{length} \\
#     B &= \text{batch} \\
#     D &= \text{NN hidden}
#     N &= \text{SSM hidden}
#   \end{aligned}
# $$


Delta_ = Float[T, '#B #L #D 1 1']
# $$
#   \begin{aligned}
#     \boldsymbol{\overline{A}} = \exp(\Delta \boldsymbol{A}) \\
#     \boldsymbol{\overline{B}} = (\Delta \boldsymbol{A})^{-1} (\boldsymbol{\overline{A}} - \boldsymbol{A}) \Delta \boldsymbol{B}
#   \end{aligned}
# $$


def discretize_zoh(A: A_, B: B_, C: C_, delta: Delta_) -> SSM:
    dA: A_ = delta * A
    A_bar: A_ = torch.exp(dA)
    # A is diagonal
    dA_inv = 1.0 / dA
    B_bar: B_ = (dA_inv * (A_bar - A) * delta) * B
    return SSM(A_bar, B_bar, C)


# $$
#   \begin{aligned}
#     (\Delta, \boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}) \mapsto (\boldsymbol{\overline{A}}, \boldsymbol{\overline{B}}, \boldsymbol{C})
#   \end{aligned}
# $$

# ## Selective SSM

# $$
#   \begin{aligned}
#     h'(t) &= \boldsymbol{A}h(t) + \boldsymbol{B}(x) x(t) \\
#     y(t) &= \boldsymbol{C}(x) h(t)
#   \end{aligned}
# $$


class Scan:
    def scan(ssm: SSM_, delta, Delta_, x: X_) -> Y_:
        selective_ssm = discretize_zoh(*ssm, delta)
        return ssm_rnn(*selective_ssm, x)


class SelectiveStructuredSSM(nn.Module):
    def __init__(self, D, N, scanner: Scan):
        init_A: Float[T, '1 1 D N 1'] = - \
            (arange(N) + 1)[None, None, None, :, None].repeat(2, D)
        self.A: A_ = torch.Parameter(init_A)
        self.s_B, self.s_C = nn.Linear(D, N), nn.Linear(D, N)
        self.s_Delta = nn.Linear(D, 1)
        self.p_Delta = torch.Parameter(torch.Tensor(D))
        self.scaner = scanner

    def forward(self, x: X_) -> Y_:
        B: Float[T, 'B L 1 N 1'] = self.s_B(x)[..., None, :, None]
        C: Float[T, 'B L 1 1 N'] = self.s_C(x)[..., None, None, :]
        ssm: SSM_ = SSM(self.A, B, C)
        Delta: Float[T, 'B L D 1 1'] = nn.softplus(self.s_Delta(x) + self.p_Delta)[..., None, None]
        return self.scanner.scan(ssm, Delta, x)


# $$
#   \begin{aligned}
#     (\Delta(x), \boldsymbol{A}, \boldsymbol{B}(x), \boldsymbol{C}(x)) \mapsto (\boldsymbol{\overline{A}}(x), \boldsymbol{\overline{B}}(x), \boldsymbol{C}(x))
#   \end{aligned}
# $$
# Full Selective


# ## Mamba Architecture
# ![](images/arch.png)

class Mamba(nn.Module):
    def __init__(self, N, D):
        self.s6 = SelectiveStructuredSSM(N, D)
        D_2 = D // 2
        self.p_up1 = nn.Linear(D_2, D)
        self.p_up2 = nn.Linear(D_2, D)
        self.p_down = nn.Linear(D, D_2)
        self.conv = nn.Conv1d()

    def forward(self, x):
        x1 = self.p_up1(x)
        x1 = torch.relu(self.conv(x1))
        x1 = self.s6(x1)
        x2 = self.p_up2(x)
        return self.p_down(x1 * torch.relu(x2))

# ## Efficient Implementation

# Mamba choices


def op(d1: tuple[A_, B_], d2: tuple[A_, B_]):
    A1, b1 = tuple(d1)
    A2, b2 = tuple(d2)
    return (A1 * A2, A1 * B_)


def pscan(op, inp):
    if inp.shape[0] == 1:
        return inp
    return pscan(
        op, op(
            [i[:, 0::2] for i in inp],
            [i[:, 1::2] for i in inp],
        ),
    )


class Scan2:
    def scan(ssm: SSM_, delta, Delta_, x: X_) -> Y_:
        pass
        # selective_ssm = discretize_zoh(*ssm, delta)

        # return ssm_rnn(*selective_ssm, x)
