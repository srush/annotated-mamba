import triton
import triton.language as tl
import torch
import math
import selective_scan_cuda
import time
ones = lambda *size: torch.ones(*size).float().cuda()
zeros = lambda *size: torch.zeros(*size).float().cuda()
arange = lambda n: torch.arange(n).float().cuda()
rand = lambda size: torch.rand(*size).abs().float().cuda()

def check(*inputs, prec=1e-4):
    for i, (a, b) in enumerate(zip(inputs[::2], inputs[1::2])):
        if isinstance(b, list):
            b = torch.tensor(b)
        c = torch.allclose(a.cpu(), b.cpu(), prec)
        c1 = torch.isclose(a.cpu(), b.cpu(), prec)
        assert c, f"{i}\n{a}\n{b}\n{c1}"
    print("match")

@triton.jit
def discretize_tt(a, b, delta):
    da = delta * a
    a_ = tl.exp(da)
    b_ = b * (a_ - 1) / da
    return a_, b_

@triton.jit
def simple_ssm_tt(X, A, B, C, Y, K: tl.constexpr):
    Ks = tl.arange(0, K)

    # Allow for a batch dimension (for Part 4)
    bid = tl.program_id(0)
    kid = bid * K
    x = tl.load(X + Ks + kid)
    a, b, c = ssm_load(Ks + kid, A, B, C)

    # Compute
    h1, h2 = tl.associative_scan((a, b*x), 0, first_order_op)
    y = c * h2

    # Save
    tl.store(Y + Ks + kid, y)

def reduce(v, rev, batch = 1):
    if rev:
        v[0, :] = v[0].flip(-1)
    o = torch.ones_like(v[0, 0])
    simple_ssm_tt[(batch,)](v[0, 1], v[0, 0], o, o, v[1, 1], K=v.shape[-1])
    v[..., -1] = 0.0
    v[:] = torch.roll(v, 1)
    if rev:
        v[1, :] = v[1].flip(-1)

@triton.jit
def select(X, mask, dim=-1):
    return tl.sum(X * mask, dim, 1)

@triton.jit
def ssm_load(Ks, A, B, C):
    "Helper for loading"
    a = tl.load(A + Ks)
    b = tl.load(B + Ks)
    c = tl.load(C + Ks)
    return a, b, c

@triton.jit
def ssm_scan(h1, h2, h2_0, reversed:tl.constexpr=0, dim:tl.constexpr=0):
    # Optional flip direction (for Part 3)
    Ks = tl.arange(0, h2.shape[dim])
    # Apply initial
    n1, n2 = first_order_op(tl.zeros_like(h1)+1.0, h2_0, h1, h2)

    # Scan
    h1, h2 = tl.associative_scan((n1, n2), dim, first_order_op, reverse=reversed)
    return h1, h2

@triton.jit
def discretize_tt(a, b, delta):
    da = delta * a
    a_ = tl.exp(da)
    b_ = b * delta
    return a_, b_

@triton.jit
def discretize_back(a, b, d, da_, db_):
    da = d * a
    a_ = tl.exp(da)

    da_da = d * a_
    da_ddelta = a * a_

    inter = (b * (da - 1) * a_ + b) / da

    #db_da = 0
    db_db = d
    db_ddelta = b

    return da_ * da_da, db_ * db_db, da_ * da_ddelta + db_ * db_ddelta


@triton.jit
def first_order_op(fl, xl, fr, xr):
    f = fr * fl
    x = fr * xl + xr
    return f, x

@triton.jit
def roll(a1, b1_last, b1_cur, a2, b2_last, b2_cur):
    return a1 + a2, tl.where(a2 == 1, b1_cur, 0) + b2_last, b2_cur

@triton.jit
def mamba_for_tt(X, dX, A, dA, B, dB, C, dC, Delta, dDelta,
             H_0, dH_0, Y, dY, H, dH,
             back:tl.constexpr,
             step:tl.constexpr,
             L: tl.constexpr, K: tl.constexpr, D_step: tl.constexpr,
             D:tl.constexpr, N: tl.constexpr):
    # Setup
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    kid = pid * K
    nH = tl.num_programs(0)
    Ba = tl.num_programs(1)
    Ks = tl.arange(0, K)[None, None, :] # 1 x 1 x K
    Ns = tl.arange(0, N)[:, None, None] # N x 1 x 1
    Nx1xK = bid*N*L + Ns*L + (Ks + kid)



    # Load forward
    b = tl.load(B + Nx1xK)
    c = tl.load(C + Nx1xK)
    db_out = tl.zeros_like(b)
    dc_out = tl.zeros_like(c)

    Ds = tl.arange(0, D_step)[None, :, None] # 1 x D x 1

    for did in range(0, D // D_step):
        DxK = bid*D*L + Ds*L + Ks + kid
        NxDx1 = bid*N*D + Ns*D + Ds
        a = tl.load(A + NxDx1)
        NxDx1_H = bid*N*D*nH + Ns*D*nH + Ds*nH + pid
        h_off = Ba*N*D*nH

        # Load forward
        delta = tl.load(Delta + DxK)
        x = tl.load(X + DxK)
        a_, b_ = discretize_tt(a, b, delta)

        if step == 2:
            h2_0 = tl.load(H_0 + 1*h_off + NxDx1_H) * (Ks == 0)
        else:
            h2_0 = tl.zeros_like(a_)
        # Compute Forward
        h1, h2 = ssm_scan(a_, b_ * x, h2_0, dim=2)
        y = tl.sum(c * h2, 0, 1)
        if step == 1:
            tl.store(H + 0 * h_off + NxDx1_H + 0*Ks, h1, Ks==K-1)
            tl.store(H + 1 * h_off + NxDx1_H + 0*Ks, h2, Ks==K-1)
        if step == 2:
            tl.store(Y + DxK, y)

        # #Compute backward
        if back == 1:
            # Load Backward
            dy = tl.load(dY + DxK)
            dh2_0 = tl.load(dH_0 + 1*h_off + NxDx1_H) * (Ks==K-1)
            delta_shift = tl.load(Delta + DxK + 1, (Ks + kid) < L - 1, 0)
            a_s, _ = discretize_tt(a, b, delta_shift)
            dh1, dh = ssm_scan(a_s, c * dy, dh2_0, reversed=1, dim=2)
            if step == 1:
                tl.store(dH + 0*h_off + NxDx1_H + 0*Ks, dh1, Ks == 0)
                tl.store(dH + 1*h_off + NxDx1_H + 0*Ks, dh, Ks == 0)

        if back == 1 and step == 2:
            dc = tl.sum(h2 * dy, 1, 1) # N x K
            _, rh2, _ = tl.associative_scan((1 + 0*(Ns + Ds + Ks), 0.0*h2, h2), 2, roll)
            rh2 = h2_0 * (Ks == 0) + rh2 * (Ks > 0)
            da, db, ddelta = discretize_back(a, b, delta, dh * rh2, dh * x)

            # Save (sums keep_dims=1)
            tl.store(dX + DxK, tl.sum(b_ * dh, 0, 1))
            tl.store(dA + NxDx1_H, tl.sum(da, 2, 1))
            tl.store(dDelta + DxK, tl.sum(ddelta, 0, 1))
            db_out = db_out + tl.sum(db, 1, 1)
            dc_out = dc_out + dc
        Ds = Ds + D_step

    if back==1 and step==2:
        tl.store(dB + Nx1xK, db_out)
        tl.store(dC + Nx1xK, dc_out)



def discretize(a, b, delta):
    da = delta * a
    a_ = torch.exp(da)
    b_ = b * delta
    return a_, b_

def mamba_torch(x, a, b, c, delta):
    "PyTorch Implementation"
    y = []
    h = 0
    a_, b_ = discretize(a, b, delta)
    for k in range(x.shape[-1]):
        h = a_[..., k] * h + b_[..., k] * x[..., k]
        y.append((c[..., k] * h).sum(1, keepdim=True))
    return h, torch.stack(y, -1)

def create(S = 128, Ba = 2, D = 4, N = 4, K=16):
    x = rand((Ba, 1, D, S))
    a = -ones((Ba, N, D, 1))
    b = ones((Ba, N, 1, S)) * 0.1
    c = rand((Ba, N, 1, S)) * 0.1
    delta = rand((Ba, 1, D, S)) * 0.1
    BLOCKS = S // K
    dx, da, db, dc, ddelta = [torch.zeros_like(b) for b in [x,a,b,c,delta]]
    da = zeros(Ba, N, D, BLOCKS)
    y, dy = [ones(Ba, 1, D, S) for _ in range(2)]
    h, dh = [zeros(2, 2, Ba, N, D, BLOCKS) for _ in range(2)]
    extra = (dx, da, db, dc, ddelta, y, dy, h, dh)
    return x, a, b, c, delta, extra

def mamba(x, a, b, c, delta, extra, K=16, D_step=2):
    #s = time.time()
    Ba = x.shape[0]
    N = a.shape[1]
    D = delta.shape[2]
    SEQLEN = x.shape[-1]
    BLOCKS = SEQLEN // K
    (dx, da, db, dc, ddelta, y, dy, h, dh) = extra
    assert BLOCKS == SEQLEN // K
    assert SEQLEN % BLOCKS == 0
    assert D % D_step == 0
    mamba_for_tt[(BLOCKS, Ba)](x, dx, a, da, b, db, c, dc, delta, ddelta, h[0], dh[0], y, dy, h[0], dh[0], back=1, step=1, L=SEQLEN, K=K, D_step=D_step, D=D, N=N)
    reduce(h, False, Ba * N * D)
    reduce(dh, True, Ba * N * D)
    mamba_for_tt[(BLOCKS, Ba)](x, dx, a, da, b, db, c, dc, delta, ddelta, h[1], dh[1], y, dy, h[1], dh[1], back=1, step=2, L=SEQLEN, K=K, D_step=D_step, D=D, N=N)
    return y, dx, da.sum(-1, keepdim=True), db, dc, ddelta


x, a, b, c, delta, extra = create()
y, dx, da, db, dc, ddelta = mamba(x, a, b, c, delta, extra, D_step=4)
for v in [x, a, b, c, delta]:
    v.requires_grad_()
_, y_ = mamba_torch(x, a, b, c, delta)
y_.sum().backward()

check(y, y_, dx, x.grad, dc, c.grad,  db, b.grad, da, a.grad, prec=1e-3)


import selective_scan_cuda
x, a, b, c, delta, extra = create(S = 8192, Ba = 8, D = 256, N=4, K=32)
mamba(x, a, b, c, delta, extra, K = 128, D_step=16)[0]

s = time.time()
for i in range(50):
    mamba(x, a, b, c, delta, extra, K = 128, D_step=16)[0]
print("TRITON:", time.time() - s)


s = time.time()
for i in range(50):
    y_them = selective_scan_cuda.fwd(x.squeeze(1), delta.squeeze(1), a[0].squeeze(-1).T, b.squeeze(-2)[:, None, :, :], c.squeeze(-2)[:, None, :, :], None, None, None, False)
print("MAMBA:", time.time() - s)
