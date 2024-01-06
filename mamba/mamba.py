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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float as F
from torch import arange, random
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

# Shape check.

# Shape

# $$
#   \begin{aligned}
#     h_t &= \boldsymbol{\overline{A}}h_t + \boldsymbol{\overline{B}}x_t \\
#     y_t &= \boldsymbol{C}h_t
#   \end{aligned}
# $$    

def ssm_step(A_bar: F[T, '... N 1'], 
             B_bar: F[T, '... N 1'],
             C_bar: F[T, '... 1 N'], 
             h: Optional[F[T, '... N 1']],
             x: F[T, '... 1 1']
             ) -> Tuple[F[T, '... N 1'], 
                        F[T, '... 1']]:
    if h is None:
        h = B_bar * x
    else:
        h = A_bar * h + B_bar * x
    y = C_bar @ h
    return h, y


# Same type

# $$
#   \begin{aligned}
#     \boldsymbol{\overline{A}} = \exp(\Delta \boldsymbol{A}) \\
#     \boldsymbol{\overline{B}} = (\Delta \boldsymbol{A})^{-1} (\boldsymbol{\overline{A}} - \boldsymbol{A}) \Delta \boldsymbol{B}
#   \end{aligned}
# $$


def discretize_zoh_diag(
        A: F[T, '... N 1'],
        B: F[T, '... N 1'],
        d: F[T, '... 1 1']
        )-> Tuple[F[T, '... N 1'], 
                  F[T, '... N 1']]:
    dA = d * A
    A_bar = torch.exp(dA)
    # A is diagonal
    dA_inv = 1.0 / dA
    B_bar = (dA_inv * (A_bar - A) * d) * B
    return A_bar, B_bar




def ssm_rnn(A_bar: F[T, '... L N 1'], 
            B_bar: F[T, '... L N 1'],
            C_bar: F[T, '... L 1 N'], 
            x: F[T, '... L 1']
            ) -> F[T, '... L 1']:
    L = x.shape[-2]
    ys = []
    h = None
    for l in range(L):
        h, y = ssm_step(A_bar[..., l, :, :], 
                        B_bar[..., l, :, :], 
                        C_bar[..., l, :, :], 
                        h, 
                        x[..., l, :, None])
        ys.append(y)
    print(ys[0].shape)
    return torch.cat(ys, dim=-2)
#
# $$
#   \begin{aligned}
#     L &= \text{length} \\
#     B &= \text{batch} \\
#     D &= \text{NN hidden}
#     N &= \text{SSM hidden}
#   \end{aligned}
# $$


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

from beartype import beartype as typechecker
from jaxtyping import jaxtyped

@dataclass
class S6:
    A : F[T, '1 D 1 N 1']
    B : F[T, 'B 1 L N 1']
    C : F[T, 'B 1 L 1 N']
    d : F[T, 'B D L 1 1']


class Scan:
    @staticmethod
    def scan(ssm: S6, x: F[T, 'B L D']) -> F[T, 'B L D']:
        A, B = discretize_zoh_diag(ssm.A, ssm.B, ssm.d)
        y = ssm_rnn(A, B, ssm.C, x.transpose(1, 2)[..., None])[..., 0]
        return y.transpose(1, 2)

def init_A(D: int, N: int) -> F[T, '1 D 1 N 1']:
    return -(arange(N) + 1).view(1, 1, 1, N, 1).expand(1, D, 1, N, 1).clone().float()

class SelectiveStructuredSSM(nn.Module):
    def __init__(self, D: int, N: int, scanner: Scan):
        super().__init__()
        self.D = D
        self.N = N        
        self.A = nn.Parameter(init_A(D=D, N=N))
        self.s_B, self.s_C = nn.Linear(D, N), nn.Linear(D, N)
        self.s_Delta = nn.Linear(D, 1)
        self.p_Delta = nn.Parameter(torch.zeros(D, 1, 1))
        self.scanner = scanner    

    def forward(self, 
                x: F[T, 'B L D']
                ) -> F[T, 'B L D']:
        assert x.shape[-1] == self.D, f"{x.shape} {self.D}"
        N = self.N
        B = self.s_B(x)[:, None, :, :, None]
        C = self.s_C(x)[:, None, :, None, :]
        d = nn.functional.softplus(self.s_Delta(x)[:, None, :, :] + self.p_Delta)[..., None]
        ssm = S6(self.A, B, C, d)
        return self.scanner.scan(ssm, x)

# $$
#   \begin{aligned}
#     (\Delta(x), \boldsymbol{A}, \boldsymbol{B}(x), \boldsymbol{C}(x)) \mapsto (\boldsymbol{\overline{A}}(x), \boldsymbol{\overline{B}}(x), \boldsymbol{C}(x))
#   \end{aligned}
# $$
# Full Selective


# ## Mamba Architecture
# ![](images/arch.png)

class Mamba(nn.Module):
    def __init__(self, N, D, scanner: Scan):
        super().__init__()
        self.s6 = SelectiveStructuredSSM(N=N, D=D, scanner=scanner)
        D_2 = D // 2
        self.p_up1 = nn.Linear(D_2, D)
        self.p_up2 = nn.Linear(D_2, D)
        self.p_down = nn.Linear(D, D_2)
        self.conv = nn.Conv1d(D, D, 5, padding=2)

    def forward(self, x: F[T, 'B L D_2']) -> F[T, 'B L D_2']:
        x1 = self.p_up1(x)
        x1 = torch.relu(self.conv(x1.transpose(1, 2))).transpose(1, 2)
        x1 = self.s6(x1)
        x2 = self.p_up2(x)
        return self.p_down(x1 * torch.relu(x2))

# ## Efficient Implementation

# Mamba choices

def main():
    B, N, D, L = 1, 2, 3, 8
    m = Mamba(N = N, D = D, scanner=Scan())
    m.forward(torch.zeros(B, L, D // 2))

main()


def op(d1: tuple[A_, B_], d2: tuple[A_, B_]):
     A1, b1 = tuple(d1)
     A2, b2 = tuple(d2)
     return (A1 * A2, A1 * B_)


def pscan(op, inp: List[F[T, "... L #A #B"]]):
    if inp.shape[] == 1:
        return inp
    return pscan(
        op, op(
            [i[..., 0::2, :, :] for i in inp],
            [i[..., 1::2, :, :] for i in inp],
        ),
    )


class Scan2:
    def scan(ssm: SSM_, delta, Delta_, x: X_) -> Y_:
        A, B = discretize_zoh_diag(ssm.A, ssm.B, ssm.d)
        y = pscan(op, [A, B, ssm.C, x.transpose(1, 2)[..., None])[..., 0]])
        return y.transpose(1, 2)
