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
import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from jaxtyping import Float as F, Integer
from torch import arange, rand
from torch import Tensor as T
from .data import *
from time import time

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
             h: F[T, '... N 1'],
             x: F[T, '...']
             ) -> F[T, '... N 1']:
    return A_bar * h + B_bar * x[..., None, None]


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
            x: F[T, '... L'],
            h_init: Optional[F[T, '... N 1']] = None
            ) -> F[T, '... L N 1']:
    L = x.shape[-1]
    h = torch.zeros_like(B_bar)
    if h_init is not None:
        h[..., -1, :, :] = h_init
    for l in range(L):
        h[..., l, :, :] = ssm_step(A_bar[..., l, :, :], 
                                   B_bar[..., l, :, :],                         
                                   h[..., l-1, :, :], 
                                   x[..., l])
        
    return h
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

@dataclass
class S6:
    A : F[T, '1 D 1 N 1']
    B : F[T, 'B 1 L N 1']
    C : F[T, 'B 1 L 1 N']
    d : F[T, 'B D L 1 1']

class Scan:
    @staticmethod
    def scan(ssm: S6, x: F[T, 'B D L'], h: Optional[F[T, 'B D N 1']]
             ) -> Tuple[F[T, 'B D L'], F[T, 'B D N 1']]:
        A, B = discretize_zoh_diag(ssm.A, ssm.B, ssm.d)
        h = ssm_rnn(A, B, x, h_init=h)
        y = ssm.C @ h
        return y.squeeze(-1).squeeze(-1), h[..., -1, :, :]

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
        self.rnn_state = self.register_buffer('rnn_state', None)
        self.cache = False

    def forward(self, x: F[T, 'B D L']) -> F[T, 'B D L']:
        assert x.shape[-2] == self.D, f"{x.shape} {self.D}"
        xT = x.transpose(1, 2)
        B = self.s_B(xT)[:, None, :, :, None]
        C = self.s_C(xT)[:, None, :, None, :]
        d = nn.functional.softplus(self.s_Delta(xT)[:, None, :, :] + self.p_Delta)[..., None]
        ssm = S6(self.A, B, C, d)
        y, h_n = self.scanner.scan(ssm, x, h=self.rnn_state if self.cache else None)
        if self.cache:
            self.rnn_state = h_n.clone()
        return y


# $$
#   \begin{aligned}
#     (\Delta(x), \boldsymbol{A}, \boldsymbol{B}(x), \boldsymbol{C}(x)) \mapsto (\boldsymbol{\overline{A}}(x), \boldsymbol{\overline{B}}(x), \boldsymbol{C}(x))
#   \end{aligned}
# $$
# Full Selective


# ## Mamba Architecture
# ![](images/arch.png)

class MambaBlock(nn.Module):
    def __init__(self, N, D, scanner: Scan):
        super().__init__()
        self.s6 = SelectiveStructuredSSM(N=N, D=D, scanner=scanner)
        D_2 = D // 2
        self.p_up1 = nn.Linear(D_2, D)
        self.p_up2 = nn.Linear(D_2, D)
        self.p_down = nn.Linear(D, D_2)
        # conv with left padding
        self.conv = nn.Conv1d(D, D, kernel_size=3, padding=0)

    def forward(self, x: F[T, 'B L D_2']) -> F[T, 'B L D_2']:
        x1 = self.p_up1(x)
        x1 = nn.functional.silu(self.conv(nn.functional.pad(x1.transpose(1, 2), (2, 0))))
        #x1 = x1.transpose(1, 2)
        x1 = self.s6(x1).transpose(1, 2)
        x2 = self.p_up2(x)
        return self.p_down(x1 * nn.functional.silu(x2))

# ## Efficient Implementation

# Mamba choices



def prepend_zero(x: F[T, '... L A B'], z : F[T, '... 1 A B']) -> F[T, "... L+1 A B"]:
    return torch.cat((z, x), dim=-3)

def evens(x: F[T, "... L A B"]) -> F[T, "... L//2 A B"]:
    return x[..., 1::2, :, :]


def odds(x: F[T, "... L A B"]) -> F[T, "... L//2 A B"]:
    return x[..., 0::2, :, :]

def split(x: F[T, "... L A B"]) -> Tuple[F[T, "... L//2 A B"], 
                                         F[T, "... L//2 A B"]]:
    return x[..., 0::2, :, :], x[..., 1::2, :, :]

def drop_first(x: F[T, "... L A B"]) -> F[T, "... L-1 A B"]:
    return x[..., 1:, :, :]

def drop_last(x: F[T, "... L A B"]) -> F[T, "... L-1 A B"]:
    return x[..., :-1, :, :]

def interleave(x: F[T, "... L A B"], y: F[T, "... L A B"]) -> F[T, "... L*2 A B"]:
    return torch.stack([x, y], dim=-3).view(*x.shape[:-3], -1, *x.shape[-2:]) 

def pscan_matrix(op,
                 inp: List[F[T, '... L _ _']], 
                 zer: List[F[T, '... 1 _ _']]
                 ) -> List[F[T, '... L _ _']]:
    L = inp[0].shape[-3] 
    assert L & (L - 1) == 0, f"{L} not power of 2"
    def pscan(x: List[F[T, '... L _ _']]) -> List[F[T, '... L+1 _ _']]:
        if x[0].shape[-3] > 1:
            evens = pscan(op(map(even, x), map(odd, x)))
            odds = op(map(drop_last, evens), map(even, x))
            x = map(interleave, odds, map(drop_first, evens))
        return list(map(prepend_zero, x, zer))
    return list(map(drop_first, pscan(inp)))


def pscan_matrix_write(op, inp: List[F[T, '... L _ _']], reverse=False):
    L = inp[0].shape[-3] 
    assert L & (L - 1) == 0, f"{L} not power of 2"
    def _op(d1, d2, write=False):
        if write:
            out = op(d2, d1)
            for d1i, d2i in zip(d1, d2): 
                d1i.copy_(d2i)
        else:
            out = op(d1, d2)
        for d2i, o in zip(d2, out):
            d2i.copy_(o)
    final = None
    def pscan(x: List[F[T, '... L _ _']]):
        nonlocal final
        if x[0].shape[-3] > 1:
            a, b = list(map(odds, x)), list(map(evens, x))
            if reverse:
                a, b = b, a
            _op(a, b)
            pscan(b)
            _op(a, b, write=True)
        else:
            final = list(map(lambda x: x.clone(), x))
            list(map(lambda x: x.zero_(), x))

    pscan(inp)
    for di, f in zip(inp, final):
        if reverse:
            di[..., 1:, :, :] = di[..., :-1, :, :].clone()
            di[..., 0, :, :] = f
        else:
            di[..., :-1, :, :] = di[..., 1:, :, :].clone()
            di[..., -1, :, :] = f

def make_scan(op, gop, start, end):
    class ParallelScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, arg1, arg2):
            ctx.save_for_backward(x, arg1, arg2)
            inp = list(map(lambda x: x.clone(), start(x, arg1)))
            pscan_matrix_write(op, inp)
            return end(inp, arg2)

        @staticmethod
        def backward(ctx, grad_output):
            x, arg1, arg2 = ctx.saved_tensors
            inp, r1 = torch.func.vjp(start, x, arg1)
            inp = list(map(lambda x: x.clone(), inp))
            pscan_matrix_write(op, inp)
            _, r2 = torch.func.vjp(end, inp, arg2)
            print(grad_output)
            grad_mid, garg2 = r2(grad_output)
            rev = list(map(lambda x: x.clone(), grad_mid))
            print(rev)
            pscan_matrix_write(gop, rev, reverse=True)
            gx, garg1 = r1(rev)
            return gx, garg1, garg2
    return ParallelScan

def test_prod():
    def _add10(d1, d2):
        return [d1[0] * d2[0], d2[0] * d1[1] + d2[1]]
    def _add10d(d1, d2):
        return [d1[0] * d2[0], d1[0] * d2[1] + d1[1]]
    x = arange(1, 5).float().clone()
    x.requires_grad_(True)
    p = (1000 * x[0]  + 100 * x[1] + 10 * x[2] + x[3])
    p.backward()
    x1g = x.grad.clone()
    x.grad.zero_()
    scan = make_scan(_add10, _add10d,
                     lambda x, _: [torch.tensor([10.0] * 4).view(4,1,1), x], 
                     lambda x, _: x[1]) 
    y = scan.apply(x.view(-1, 1, 1), torch.tensor([]), torch.tensor([]))
    y.sum().backward()
    assert (x.grad == x1g).all(), f"{x.grad} {x1g}"

test_prod()

def test_add2():
    def _add(d1, d2):
        return [d1[0] + d2[0]]
    x = arange(16).float().clone()
    x.requires_grad_(True)
    x.view(-1).cumsum(0).sum().backward()
    x1g = x.grad.clone()
    x.grad.zero_()
    scan = make_scan(_add, lambda x, _: [x], lambda x, _: x[0]) 
    y = scan.apply(x.view(-1, 1, 1), torch.tensor([]), torch.tensor([]))
    y.sum().backward()
    print(x.grad)
    assert (x.grad == x1g).all()
    assert (x.view(-1).cumsum(0) == y.view(-1)).all(), (x.view(-1).cumsum(0), y.view(-1)) 

test_add2()


def test_add():
    def _add(d1, d2):
        return list(map(lambda a, b: a + b, d1, d2))
    x = arange(16).view(-1, 1, 1).float()
    y = pscan_matrix(_add, [x], [torch.zeros(1, 1, 1).float()])
    assert (x.view(-1).cumsum(0) == y[0].view(-1)).all(), (x.cumsum(0), y[0].view(-1)) 
test_add()

DT = F[T, '... N 1']
SSMHid = Tuple[DT, DT]

def ssm_merge(d1, d2):
    A1, b1 = d1
    A2, b2 = d2
    return [A1 * A2, A2 * b1 + b2]


def init_scan(B: F[T, '... L N 1'], x: F[T, '... L']) -> F[T, '... L N 1']:
    return B * x[..., None, None]

def test_scan():
    L = 8
    N = 2
    A, B, C = rand(L, N, 1), rand(L, N, 1), rand(L, 1, N)
    x = rand(L)
    h = ssm_rnn(A, B, x)
    y1 = C @ h
    
    b1 = init_scan(B, x)
    _, h = pscan_matrix(ssm_merge, [A, b1], [torch.zeros(*A.shape[:-3], 1, *A.shape[-2:]), 
                                             torch.zeros(*b1.shape[:-3], 1, *b1.shape[-2:])])
    y2 = C @ h
    assert torch.isclose(y1, y2).all(), f"{y1.squeeze()} {y2.squeeze()}"
  
test_scan()


class ParallelScan(Scan):
    @staticmethod
    def scan(ssm: S6, x: F[T, 'B D L'], h: Optional[F[T, 'B D N 1']]
             ) -> Tuple[F[T, 'B D L'], F[T, 'B D N 1']]:
        assert h is None
        
        A, B = discretize_zoh_diag(ssm.A, ssm.B, ssm.d)
        b1 = init_scan(B, x)
        _, h = pscan_matrix(ssm_merge, [A, b1], [torch.zeros(*A.shape[:-3], 1, *A.shape[-2:], device=A.device), 
                                                 torch.zeros(*b1.shape[:-3], 1, *b1.shape[-2:], device=b1.device)])
        y = ssm.C @ h
        return y[..., 0, 0], h[..., -1, :, :]

def ssm_merge2(A):
    def inner(d1, d2):
        delta1, b1 = d1
        delta2, b2 = d2
        return [delta1 + delta2, torch.exp(A * delta2) * b1 + b2]
    return inner

class ParallelScan2(Scan):
    @staticmethod
    def scan(ssm: S6, x: F[T, 'B D L'], h: Optional[F[T, 'B D N 1']]
             ) -> Tuple[F[T, 'B D L'], F[T, 'B D N 1']]:
        assert h is None
        d = ssm.d
        _, B = discretize_zoh_diag(ssm.A, ssm.B, ssm.d)
        b1 = init_scan(B, x)
        _, h = pscan_matrix(ssm_merge2(ssm.A), [ssm.d, b1], [torch.zeros(*d.shape[:-3], 1, *d.shape[-2:], device=d.device), 
                                                             torch.zeros(*b1.shape[:-3], 1, *b1.shape[-2:], device=b1.device)])
        y = ssm.C @ h
        return y[..., 0, 0], h[..., -1, :, :]

def test_full():
    B, N, D, L = 1, 2, 3, 8
    m = MambaBlock(N = N, D = D, scanner=ParallelScan2())
    out = m.forward(torch.ones(B, L, D // 2))
    m.scanner = Scan()
    out2 = m.forward(torch.ones(B, L, D // 2))
    print("test_full")
    assert torch.isclose(out, out2).all(), f"{out} {out2}"
test_full()

# Model with multiple stacked mamba blocks that starts with embeddings and produces a softmax output. 
# Uses module list to store the mamba blocks. 
# __init__ Takes number of input and output classes, layers, N, D and scanner as args.
class Mamba(nn.Module):
    def __init__(self, 
                 N: int, 
                 D: int, 
                 scanner: Scan,
                 layers: int = 1, 
                 n_classes: int = 10):
        super().__init__()
        self.emb = nn.Embedding(n_classes, D // 2)
        self.mamba_blocks = nn.ModuleList([MambaBlock(N, D, scanner) for _ in range(layers)])
        self.predict = nn.Linear(D // 2, n_classes)

    def cache(self, cache: bool = True):
        for mamba_block in self.mamba_blocks:
            mamba_block.s6.cache = cache
            mamba_block.s6.scanner = Scan()

    def forward(self, x: Integer[T, 'B L 1']) -> F[T, 'B L C']:
        x = self.emb(x).squeeze(-2)
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        x = self.predict(x)
        return x

    # Code to sample one step from the model as if it were an RNN. Keep a cache. 


# The code for training and test a sequence model with an adam optimizer. Takes train and test loaders and a model as input.

def train(model: nn.Module, 
          trainloader: torch.utils.data.DataLoader, 
          epochs: int = 10, 
          lr: float = 0.001):
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        running_loss = 0.0
        start = time()
        for i, (x, _) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            L = x.shape[1]
            goal = 2 ** math.ceil(math.log2(L))
            out = model(torch.cat([x.long(), torch.zeros(x.shape[0], goal-L, 1)], 1).long().cuda())
            loss = criterion(out[:, :-1].contiguous().view(-1, out.shape[-1]), 
                             torch.cat([x[:, 1:, 0], torch.zeros(x.shape[0], goal-L) - 100], 1).long().view(-1).cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 20:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100} time: {time() - start}")
                start = time()
                running_loss = 0.0
                return
def generate(model: nn.Module, testloader: torch.utils.data.DataLoader):
    for i, (x, y) in enumerate(testloader, 0):
        batch, seq = x.shape[:2]
        x = x[:, :2, :].long().cuda()
        # Generate 128 outputs starting from x[0]
        model.cache()
        model.eval()
        model(x[:1, :1, :])
        with torch.no_grad():
            for _ in range(10):
                next = model(x[:1, -1:, :])
                next_word = next.argmax(-1)
                x = torch.cat((x, next_word[..., None]), dim=-2)
            print(x)

#trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = create_sin_x_dataset(bsz=16)
trainloader, testloader, N_CLASSES, SEQ_LENGTH, IN_DIM = create_mnist_dataset(bsz=16)
m = Mamba(16, 128, ParallelScan2(), 5, N_CLASSES)
m.cuda()
from torch.profiler import profile, record_function, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        train(m, trainloader, epochs=1)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
#generate(m, testloader)

