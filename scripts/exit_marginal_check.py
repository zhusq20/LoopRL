"""Exact-enumeration check of the exit-marginal gradient (scheme 2) under full-state vs dynamic early exit.

Backs two claims in the design notes (scheme 2) and README section 8. Run: python scripts/exit_marginal_check.py
Needs only torch; runs on CPU in a few seconds.

Check 1. With Ouro's per-loop gate (lambda_t = sigmoid(a_t), q_t = lambda_t * prod_{s<t}(1 - lambda_s),
last loop takes the remaining mass), the reward gradient on the gate logit is
    d log pi_mix(y) / d a_t = rho_{>=t} * (rho_t / rho_{>=t} - lambda_t)
and it vanishes when all loops agree on y.

Check 2. The exact policy gradient of a process with early exit decomposes as
    E[A * sum_j grad log pi_mix(y_j | s_j)]  +  E[A * sum_j grad log rho(t_j | s_j, y_j)]
The first term is scheme 2's gradient. The second is zero in expectation when every position runs
all loops, and nonzero once the exit depth changes what later tokens can see.

Toy process: 2 token positions, K=4 exits, V=3 tokens. Position 2's gate and readouts
depend on (y1, e1), where e1 is what later tokens can see of token 1's depth.
  full-state process : e1 = K (all loops always visible)  -> exit does not affect the future
  dynamic process    : e1 = t1 (only loops <= exit visible) -> "memory channel", strength m
Everything is enumerated exactly (144 paths), so there is no sampling noise.

Expected output: check 1 differs by ~1e-17. In check 2 the "bias" column is ~1e-16 for the full-state
process and for m = 0, grows with m for the dynamic process, equals the position-1 correction, and the
residual after adding the correction is ~1e-15. The position-2 (last position) correction is always ~0.
The bias percentages are properties of this toy process, not estimates for Ouro.
"""
import itertools
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
K, V = 4, 3


def exit_dist(a):
    """Stick-breaking: lambda_t = sigmoid(a_t), q_t = lambda_t * prod_{s<t}(1-lambda_s); last exit takes the rest."""
    lam = torch.sigmoid(a)
    surv = torch.cat([torch.ones(1), torch.cumprod(1 - lam, 0)])
    return torch.cat([lam * surv[:-1], surv[-1:]]), lam


def make_params():
    p = dict(
        a1=torch.randn(K - 1), L1=torch.randn(K, V),
        a2_base=torch.randn(V, K - 1), L2_base=torch.randn(V, K, V),
        a2_mem=torch.randn(V, K, K - 1), L2_mem=torch.randn(V, K, K, V),
    )
    for v in p.values():
        v.requires_grad_(True)
    return p


def pos2(p, y1, e1, m):
    # e1 == K-1 (index of the deepest exit) is the "everything visible" state: no memory offset there
    off = 0.0 if e1 == K - 1 else m
    q, lam = exit_dist(p["a2_base"][y1] + off * p["a2_mem"][y1, e1])
    P = torch.softmax(p["L2_base"][y1] + off * p["L2_mem"][y1, e1], -1)
    return q, P


def enumerate_paths(p, m, dynamic):
    """Yield (prob, y1, y2, log_mix_sum, log_rho_1, log_rho_2) for every path."""
    q1, _ = exit_dist(p["a1"])
    P1 = torch.softmax(p["L1"], -1)
    for t1, y1 in itertools.product(range(K), range(V)):
        mix1 = (q1 * P1[:, y1]).sum()
        rho1 = q1[t1] * P1[t1, y1] / mix1
        q2, P2 = pos2(p, y1, t1 if dynamic else K - 1, m)
        for t2, y2 in itertools.product(range(K), range(V)):
            mix2 = (q2 * P2[:, y2]).sum()
            rho2 = q2[t2] * P2[t2, y2] / mix2
            prob = q1[t1] * P1[t1, y1] * q2[t2] * P2[t2, y2]
            yield prob, y1, y2, mix1.log() + mix2.log(), rho1.log(), rho2.log()


def flat_grad(scalar, p):
    g = torch.autograd.grad(scalar, list(p.values()), allow_unused=True, retain_graph=True)
    return torch.cat([torch.zeros_like(v).flatten() if gi is None else gi.flatten() for gi, v in zip(g, p.values())])


def run(m, dynamic, R):
    p = make_params()
    J = marg = corr1 = corr2 = total = 0.0
    for prob, y1, y2, log_mix, lr1, lr2 in enumerate_paths(p, m, dynamic):
        w = prob.detach() * R[y1, y2]
        J = J + prob * R[y1, y2]          # true objective, differentiated exactly
        marg = marg + w * log_mix         # scheme 2: A * grad log pi_mix, per token
        corr1 = corr1 + w * lr1           # A * grad log rho_{t1}
        corr2 = corr2 + w * lr2           # A * grad log rho_{t2}  (last position)
        total = total + prob.detach()
    g_true, g_marg = flat_grad(J, p), flat_grad(marg, p)
    g_c1, g_c2 = flat_grad(corr1, p), flat_grad(corr2, p)
    return dict(
        total=float(total),
        bias=float((g_true - g_marg).norm() / g_true.norm()),
        resid=float((g_true - g_marg - g_c1 - g_c2).norm() / g_true.norm()),
        c1=float(g_c1.norm() / g_true.norm()),
        c2=float(g_c2.norm() / g_true.norm()),
    )


# 1) hazard-form gate gradient: d log pi_mix(y) / d a_t = rho_{>=t} * (rho_t / rho_{>=t} - lambda_t)
a = torch.randn(K - 1, requires_grad=True)
P = torch.softmax(torch.randn(K, V), -1)
q, lam = exit_dist(a)
y = 1
mix = (q * P[:, y]).sum()
(auto,) = torch.autograd.grad(mix.log(), a)
rho = (q * P[:, y] / mix).detach()
tail = rho.flip(0).cumsum(0).flip(0)                      # rho_{>=t}
closed = tail[:-1] * (rho[:-1] / tail[:-1] - lam.detach())
print("1) hazard gate gradient, autograd vs closed form, max abs diff:", float((auto - closed).abs().max()))
P_same = torch.softmax(torch.randn(V), -1).expand(K, V)   # all loops agree
mix_same = (exit_dist(a)[0] * P_same[:, y]).sum()
print("   all loops agree -> gate gradient norm:", float(torch.autograd.grad(mix_same.log(), a)[0].norm()))

# 2) full-state vs dynamic process
R = torch.rand(V, V)
print("\n2) relative error of scheme 2's gradient  |g_true - g_marg| / |g_true|")
print("   process              m     bias      corr(pos1)  corr(pos2)  residual after adding corr")
for dynamic, m in [(False, 1.0), (True, 0.0), (True, 0.1), (True, 0.3), (True, 1.0)]:
    torch.manual_seed(1)
    r = run(m, dynamic, R)
    name = "dynamic (ragged)" if dynamic else "full-state      "
    print(f"   {name}  {m:4.1f}   {r['bias']:.2e}  {r['c1']:.2e}    {r['c2']:.2e}    {r['resid']:.2e}   (sum p = {r['total']:.6f})")
