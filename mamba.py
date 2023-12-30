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


import torch
from torch import Tensor
import torch.nn as nn



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


@dataclass
class SSM:
    A: Tensor
    B: Tensor
    C: Tensor
    delta: Tensor
    def unpack(self):
        return self.A, self.B, self.C



@dataclass
class DiscreteSSM:
    A_bar: Tensor
    B_bar: Tensor
    C: Tensor
    def unpack(self):
        return self.A_bar, self.B_bar, self.C


# $$
#   \begin{aligned}
#     h_t &= \boldsymbol{\overline{A}}h_t + \boldsymbol{\overline{B}}x_t \\
#     y_t &= \boldsymbol{C}h_t 
#   \end{aligned}
# $$


def ssm_rnn(ssm: DiscreteSSM, x: Tensor):
    A_bar, B_bar, C = ssm.unpack()
    h_t_1 = torch.zeros_like(C)
    for x_t in x:
        h_t = A_bar @ h_t_1 + B_bar @ x_t
        y_t = C @ h
        yield y_t
        h_t_1 = h_t
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
#     \boldsymbol{\overline{A}} = \exp(\Delta \boldsymbol{A}) \\
#     \boldsymbol{\overline{B}} = (\Delta \boldsymbol{A})^{-1} (\boldsymbol{\overline{A}} - \boldsymbol{A}) \Delta \boldsymbol{B}
#   \end{aligned}
# $$
def discretize_zoh(ssm: SSM) -> DiscreteSSM:
    A, B, C, delta = ssm.unpack()
    A_bar = torch.exp(delta * A)
    B_bar = torch.inverse(delta * A) @ (A_bar - A) @ delta * B    
    return DiscreteSSM(A_bar, B_bar, C)


# $$
#   \begin{aligned}
#     (\Delta, \boldsymbol{A}, \boldsymbol{B}, \boldsymbol{C}) \mapsto (\boldsymbol{\overline{A}}, \boldsymbol{\overline{B}}, \boldsymbol{C})
#   \end{aligned}
# $$


# Structured State Space. 
# D x N S4D matrix. 

class StructuredMatrix:
    def __init__(self, rep: Tensor):
        self._rep = rep

    def s4d_real(shape)
        N = shape[-1]
        # fix up
        return -(torch.arange(N) + 1).view(N).reshape(shape)
    
    def __matmul__(self, other: Tensor):
        # Diagonal mult
        return None

# ## Selective SSM

# $$
#   \begin{aligned}
#     h'(t) &= \boldsymbol{A}h(t) + \boldsymbol{B}(x) x(t) \\
#     y(t) &= \boldsymbol{C}(x) h(t) 
#   \end{aligned}
# $$
    

class SelectiveSSM:
    def __init__(self, D, N):
        self.D = D
        self.N = N
        self.A = torch.Parameter()
        self.s_B = torch.nn.Linear(D, N)
        self.s_C = torch.nn.Linear(D, N)
        self.s_Delta = torch.linear(D, 1)
        self.p_Delta = torch.Parameter(torch.Tensor(D))

    def forward(self, x: Tensor):
        BATCH, L, D = = x.shape
        assert D == self.D
        B = self.s_B(x) # B, L, N
        C = self.s_C(x) # B, L, N
        Delta = torch.nn.softplus(self.s_Delta(x) + self.p_Delta) # B, L, D
        return SSM(self.A, B, C, Delta)


# $$
#   \begin{aligned}
#     (\Delta(x), \boldsymbol{A}, \boldsymbol{B}(x), \boldsymbol{C}(x)) \mapsto (\boldsymbol{\overline{A}}(x), \boldsymbol{\overline{B}}(x), \boldsymbol{C}(x))
#   \end{aligned}
# $$
# Full Selective     

class S6:
    def __init__(self, N: int, D:int) -> None:
        self.selective_ssm = SelectiveSSM(N, D)

    def forward(self, x):
        selective_ssm: SelectiveSSM = self.selective_ssm(x)
        discrete_ssm = discretize_zoh(selective_ssm)
        y = discrete_ssm(x)
        return y

# ## Mamba Architecture
    
# 
# 
    
# ![](images/arch.png)

class Mamba():
    def __init__(self, N, D):
        D_2 = D / 2
        self.s6 = S6(N, D)
        self.p_up1 = nn.Linear(D_2, D)
        self.p_up2 = nn.Linear(D_2, D)
        self.p_down = nn.Linear(D, D_2)
        self.conv = nn.Conv1d()
        
    def forward(self, x):
        sigma = torch.relu
        x1 = self.p_up1(x)
        x1 = sigma(self.conv(x1))
        x1 = self.s6(x1)
        x2 = self.p_up2(x)
        return self.p_down(x1 * sigma(x2))

# ## Efficient Implementation

# Mamba choices


