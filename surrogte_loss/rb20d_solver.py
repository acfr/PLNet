import os 
import jax 
from jax import jacfwd, jacrev
import jax.numpy as jnp
from typing import Sequence, Callable
import scipy.io 
import matplotlib.pyplot as plt
from layer import *
import orbax.checkpoint
import numpy as np
import jax.random as random
from rosenbrock_utils import Sampler, Rosenbrock

class MonLipNet(nn.Module):
    units: Sequence[int]
    tau: jnp.float32 = 10.
    # mu: jnp.float32 = 0.1 # Monotone lower bound
    # nu: jnp.float32 = 10.0 # Lipschitz upper bound (nu > mu)
    # act_fn: Callable = nn.relu

    def get_bounds(self):
        lognu = self.variables['params']['lognu']
        nu = jnp.squeeze(jnp.exp(lognu), 0)
        mu = nu / self.tau

        return mu, nu, self.tau
    
    def get_params(self):
        lognu = self.variables['params']['lognu']
        nu = jnp.squeeze(jnp.exp(lognu), 0)
        mu = nu / self.tau
        Fq = self.variables['params']['Fq']
        fq = self.variables['params']['fq']
        Q = cayley((fq / jnp.linalg.norm(Fq)) * Fq).T 
        V, S, bh = [], [], []
        idx = 0
        L = len(self.units)
        for k, nz in zip(range(L), self.units):
            Qk = Q[idx:idx+nz, :] 
            b = self.variables['params'][f'b{k}']
            bh.append(b)
            Fab = self.variables['params'][f'Fab{k}']
            fab = self.variables['params'][f'fab{k}']
            ABT = cayley((fab / jnp.linalg.norm(Fab)) * Fab)
            if k > 0:
                Ak, Bk = ABT[:nz, :].T, ABT[nz:, :].T
                V.append(2 * Bk @ ATk_1)
                S.append(Ak @ Qk - Bk @ Qk_1)
            else:
                Ak = ABT.T      
                S.append(ABT.T @ Qk)
            
            ATk_1, Qk_1 = Ak.T, Qk
            idx += nz

        by = self.variables['params']['by']
        bh = jnp.concatenate(bh, axis=0)
        S = jnp.concatenate(S, axis=0)

        params = {
            "mu": mu,
            "gam": nu - mu,
            "units": self.units,
            "V": V, 
            "S": S,
            "by": by,
            "bh": bh
        }

        return params

    @nn.compact
    def __call__(self, x : jnp.array) -> jnp.array:
        nx = jnp.shape(x)[-1]  
        lognu = self.param('lognu', nn.initializers.constant(jnp.log(2.)), (1,), jnp.float32)
        nu = jnp.exp(lognu)
        mu = nu / self.tau 
        by = self.param('by', nn.initializers.zeros_init(), (nx,), jnp.float32) 
        y = mu * x + by 
        
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (nx, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq) 
        sqrt_2g, sqrt_g2 = jnp.sqrt(2. * (nu - mu)), jnp.sqrt((nu - mu) / 2.)
        idx, nz_1 = 0, 0 
        zk = x[..., :0]
        Ak_1 = jnp.zeros((0, 0))
        for k, nz in enumerate(self.units):
            Fab = self.param(f'Fab{k}', nn.initializers.glorot_normal(), (nz+nz_1, nz), jnp.float32)
            fab = self.param(f'fab{k}',nn.initializers.constant(jnp.linalg.norm(Fab)), (1,), jnp.float32)
            ABT = cayley((fab / jnp.linalg.norm(Fab)) * Fab)
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]
            QTk_1, QTk = QT[:, idx-nz_1:idx], QT[:, idx:idx+nz]
            STk = QTk @ ATk - QTk_1 @ BTk 
            bk = self.param(f'b{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            # use relu activation, no need for psi
            # pk = self.param(f'p{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            zk = nn.relu(2 * (zk @ Ak_1) @ BTk + sqrt_2g * x @ STk + bk)
            # zk = nn.relu(zk * jnp.exp(-pk)) * jnp.exp(pk)
            y += sqrt_g2 * zk @ STk.T  
            idx += nz 
            nz_1 = nz 
            Ak_1 = ATk.T     

        return y 
        
class BiLipNet(nn.Module):
    units: Sequence[int]
    tau: jnp.float32
    depth: int = 2

    def setup(self):
        uni, mon = [], []
        layer_tau = (self.tau) ** (1/self.depth)
        for _ in range(self.depth):
            uni.append(Unitary())
            mon.append(MonLipNet(self.units, tau=layer_tau))
        uni.append(Unitary())
        self.uni = uni
        self.mon = mon

    def get_bounds(self):
        lipmin, lipmax, tau = 1., 1., 1.
        for k in range(self.depth):
            mu, nu, ta = self.mon[k].get_bounds()
            lipmin *= mu 
            lipmax *= nu 
            tau *= ta 
        return lipmin, lipmax, tau 
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        for k in range(self.depth):
            x = self.uni[k](x)
            x = self.mon[k](x)
        x = self.uni[self.depth](x)
        return x 

class PLNet(nn.Module):
    BiLipBlock: nn.Module

    def gmap(self, x: jnp.array) -> jnp.array:
        return self.BiLipBlock(x)

    def get_bounds(self):
        return self.BiLipBlock.get_bounds()
    
    def vgap(self, x: jnp.array) -> jnp.array:
        y = self.BiLipBlock(x)
        return 0.5 * (jnp.linalg.norm(y, axis=-1) ** 2)
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        x = self.BiLipBlock(x)
        y = QuadPotential()(x)

        return y 
## mln utils -------------------------------------------------------------
def orth_fwd(params, x):
    return x @ params['R'].T + params['b'] 

def orth_bwd(params, y):
    return (y - params['b']) @ params['R']

def mln_fwd(params, x):
    b = mln_fwd_x2b(params, x)
    z = mln_fwd_b2z(params, b)
    y = mln_fwd_xz2y(params, x, z)

    return y

def mln_fwd_b2z(params, b):
    z = []
    idx = 0
    for k, nz in enumerate(params['units']):
        if k == 0:
            zk = nn.relu(b[..., idx:idx+nz])
        else:
            zk = nn.relu(zk @ params['V'][k-1].T + b[..., idx:idx+nz])
        z.append(zk)
        idx += nz 
    return jnp.concatenate(z, axis=-1)

def mln_fwd_xz2y(params, x, z):
    return params['mu'] * x + jnp.sqrt(params['gam']/2) * z @ params['S'] + params['by']

def mln_bwd_yz2x(params, y, z):
    return (y - params['by'] - jnp.sqrt(params['gam']/2) * z @ params['S']) / params['mu'] 

def mln_fwd_x2b(params, x):
    return jnp.sqrt(2*params['gam']) * x @ params['S'].T + params['bh']

def mln_bwd_y2b(params, y):
    return jnp.sqrt(2*params['gam'])/params['mu'] * (y-params['by']) @ params['S'].T + params['bh']

def mln_bwd_z2v(params, z):
    return params['gam']/params['mu'] * (z @ params['S']) @ params['S'].T 

def mln_RA(params, alpha, bz, zh, uh):
    zv = mln_bwd_z2v(params, zh)
    vh = bz - zv 
    au, av = 1/(1+alpha), alpha/(1+alpha)
    b = av * vh + au * uh
    z = []
    idx = 0
    for k, nz in enumerate(params['units']):
        if k == 0:
            zk = b[..., idx:idx+nz]
        else:
            zk = av * zk @ params['V'][k-1].T + b[..., idx:idx+nz]
        z.append(zk)
        idx += nz 
    return jnp.concatenate(z, axis=-1)

# solver -----------------------------------------------------------
def adam_solver(
    fn: Callable,
    Lr: float, 
    z0: jnp.array, 
    max_iter: int = 500
):

    grad_fn = jax.jit(jax.vmap(jax.value_and_grad(lambda z: fn(z))))

    # implementation of adam 
    vgap = []
    mt = jnp.zeros_like(z0)
    vt = jnp.linalg.norm(mt, axis=-1, keepdims=True)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    beta1t, beta2t = beta1, beta2
    clip_bound = 10.0
    for k in range(max_iter):
        v, gt = grad_fn(z0)
        vg = jnp.mean(v)
        vgap.append(vg)
        gt = jnp.clip(gt, a_min=-clip_bound, a_max=clip_bound)
        mt = beta1 * mt + (1-beta1) * gt 
        vt = beta2 * vt + (1-beta2) * jnp.linalg.norm(gt, axis=-1, keepdims=True) ** 2
        mht = mt / (1 - beta1t)
        vht = vt / (1 - beta2t)
        beta1t *= beta1 
        beta2t *= beta2 
        z0 -= Lr * mht / (jnp.sqrt(vht) + eps)
        print(f'Iter. {k:4d} | v: {vg:.8f}')

    data = {
        'vgap': jnp.array(vgap),
        'step': jnp.array([i for i in range(max_iter)]),
        'z': z0,
        'alpha': Lr
    }

    return data

def mln_back_solve_fwd(params, z, fn,
    max_iter: int = 500,
    tau: float = 10.0
):
    ltau = jnp.sqrt(tau)
    orth = Unitary()
    mln = MonLipNet([256]*8, tau=ltau)

    block_param = params['params']['BiLipBlock']
    uni_params, mon_params = [], []
    depth = 2
    for k in range(depth):
        p = orth.apply({'params': block_param[f'uni_{k}']}, method=orth.get_params)
        uni_params.append(p)
        p = mln.apply({'params': block_param[f'mon_{k}']}, method=mln.get_params)
        mon_params.append(p)
    p = orth.apply({'params': block_param[f'uni_{depth}']}, method=orth.get_params)
    uni_params.append(p) 

    Mu = mon_params[0]['mu']
    Nu = Mu + mon_params[0]['gam']
    alpha0 = Mu / (Nu ** 2)
    
    Mu = mon_params[1]['mu']
    Nu = Mu + mon_params[1]['gam']
    alpha1 = Mu / (Nu ** 2)

    def cond_fn(state):
        _, iter, iter_max, _ = state
        return iter < iter_max
            
    def body1_fn(state):
        x, iter, iter_max, y = state
        x -= alpha1 * (mln_fwd(mon_params[1], x) - y)
        return (x, iter+1, iter_max, y)
    
    def body0_fn(state):
        x, iter, iter_max, y = state
        x -= alpha0 * (mln_fwd(mon_params[0], x) - y)
        return (x, iter+1, iter_max, y)
    
    vgap, step = [], []
    shp = jnp.shape(z)
    zopt = jnp.zeros(shp)
    k = 0
    while k <= max_iter:
        if k == 0:
            z0 = z
        else:
            y = orth_bwd(uni_params[2], zopt)
            x = jax.lax.while_loop(cond_fn, body1_fn, (z, 0, k, y))[0]
            y = orth_bwd(uni_params[1], x)
            x = jax.lax.while_loop(cond_fn, body0_fn, (z, 0, k, y))[0]
            z0 = orth_bwd(uni_params[0], x)
        v = jnp.mean(fn(z0))
        vgap.append(v)
        step.append(k)
        print(f'Iter. {k:4d} | v: {v:.8f}')
        if k <= 200:
            k += 1
        else:
            k += 10

    data = {
        'vgap': jnp.array(vgap),
        'step': jnp.array(step),
        'z': z0
    }

    return data 

def mln_back_solve_dys(params, z, fn,
    max_iter: int = 500,
    tau: float = 10.0
):
    ltau = jnp.sqrt(tau)
    orth = Unitary()
    mln = MonLipNet([256]*8, tau=ltau)

    block_param = params['params']['BiLipBlock']
    uni_params, mon_params = [], []
    depth = 2
    for k in range(depth):
        p = orth.apply({'params': block_param[f'uni_{k}']}, method=orth.get_params)
        uni_params.append(p)
        p = mln.apply({'params': block_param[f'mon_{k}']}, method=mln.get_params)
        mon_params.append(p)
    p = orth.apply({'params': block_param[f'uni_{depth}']}, method=orth.get_params)
    uni_params.append(p) 

    alpha0 = alpha*mon_params[0]['mu'] / mon_params[0]['gam']
    alpha1 = alpha*mon_params[1]['mu'] / mon_params[1]['gam']
    
    def DavisYinSplit(params, uk, bz, alpha):
        zh = nn.relu(uk)
        uh = 2*zh - uk 
        zk = mln_RA(params, alpha, bz, zh, uh)
        uk += Lambda * (zk - zh) 

        return zk, uk

    def cond_fn(state):
        _, iter, iter_max, _, _ = state
        return iter < iter_max
            
    def body1_fn(state):
        _, iter, iter_max, uk, bz = state
        zk, uk = DavisYinSplit(mon_params[1], uk, bz, alpha1)
        return (zk, iter+1, iter_max, uk, bz)
    
    def body0_fn(state):
        _, iter, iter_max, uk, bz = state
        zk, uk = DavisYinSplit(mon_params[0], uk, bz, alpha0)
        return (zk, iter+1, iter_max, uk, bz)
    
    vgap, step = [], []
    shp = jnp.shape(z)
    zopt = jnp.zeros(shp)
    k = 0
    while k <= max_iter:
        if k == 0:
            z0 = z
        else:
            y = orth_bwd(uni_params[2], zopt)
            bz = mln_bwd_y2b(mon_params[1], y)
            uk = jnp.zeros(jnp.shape(bz))
            zk = jax.lax.while_loop(cond_fn, body1_fn, (uk, 0, k, uk, bz))[0]
            x = mln_bwd_yz2x(mon_params[1], y, zk)
            y = orth_bwd(uni_params[1], x)
            bz = mln_bwd_y2b(mon_params[0], y)
            uk = jnp.zeros(jnp.shape(bz))
            zk = jax.lax.while_loop(cond_fn, body0_fn, (uk, 0, k, uk, bz))[0]
            x = mln_bwd_yz2x(mon_params[0], y, zk)
            z0 = orth_bwd(uni_params[0], x)
        v = jnp.mean(fn(z0))
        vgap.append(v)
        step.append(k)
        print(f'Iter. {k:4d} | v: {v:.8f}')
        if k <= 200:
            k += 1
        else:
            k += 10
    
    data = {
        'vgap': jnp.array(vgap),
        'step': jnp.array(step),
        'z': z0
    }

    return data         

root_dir = './results'
max_iter = 50
rng = random.PRNGKey(43)
z = Sampler(rng, 10000, 20)

# # Run solvers for BiLipNet ---------------------------------------------
name = "BiLipNet"
depth = 2
alpha = 1.0
Lambda = 1.0

for tau in [5, 50]:
    model = PLNet(BiLipNet([256]*8, depth=depth, tau=tau))
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    train_dir = f'{root_dir}/rosenbrock-dim20-batch50/{name}-{depth}-tau{tau}'
    params = orbax_checkpointer.restore(f'{train_dir}/ckpt/params')
    fn = lambda x : model.apply(params, x, method=model.vgap)

    # for lr in [1.0, 2.0, 5.0, 10.0]:
    #     data = adam_solver(fn, lr, z, max_iter=max_iter)
    #     fig, axes = plt.subplots(1,1)
    #     axes.semilogy(data['step'], data['vgap'])
    #     plt.savefig(f'{train_dir}/adam-lr{lr:.2f}.pdf')
    #     plt.close()
    #     scipy.io.savemat(f'{train_dir}/adam-lr{lr:.2f}.mat', data)
        
    # data = mln_back_solve_fwd(params, z, fn, max_iter=max_iter, tau=tau)
    # plt.semilogy(data['step'], data['vgap'])
    # plt.savefig(f'{train_dir}/FWD-PLNet.pdf')
    # plt.close()
    # scipy.io.savemat(f'{train_dir}/FWD-PLNet.mat', data)

    data = mln_back_solve_dys(params, z, fn, max_iter=max_iter, tau=tau)
    plt.semilogy(data['step'], data['vgap'])
    plt.savefig(f'{train_dir}/DYS-PLNet-alpha{alpha:.1f}-lambda{Lambda:.1f}.pdf')
    plt.close()
    scipy.io.savemat(f'{train_dir}/DYS-PLNet-alpha{alpha:.1f}-lambda{Lambda:.1f}.mat', data)


# plots -----------------------------------------------------------------------------
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
name = 'BiLipNet'
depth = 2
dataset = 'rosenbrock'

tau=5
train_dir = f'./results/rosenbrock-dim20-batch50/{name}-{depth}-tau{tau}'

lr = 2.0
gd_mln = scipy.io.loadmat(f'{train_dir}/adam-lr{lr:.2f}.mat')
bs_fwd = scipy.io.loadmat(f'{train_dir}/FWD-PLNet.mat')
bs_dys = scipy.io.loadmat(f'{train_dir}/DYS-PLNet.mat')

ax = axes[0]
ax.semilogy(bs_dys['step'][0, :], bs_dys['vgap'][0, :], linewidth=2, label=r'DYS$\left(\frac{\mu}{\nu-\mu}\right)$')
ax.semilogy(bs_fwd['step'][0, :], bs_fwd['vgap'][0, :], linewidth=2, label=r'FSM$\left(\frac{\mu}{\nu^2}\right)$')
ax.semilogy(gd_mln['step'][0, :], gd_mln['vgap'][0, :], linewidth=2, label='ADAM(2.0)')

ax.set_xlabel('Steps')
ax.set_xlim(0, 100)
ax.set_ylabel('Surrogate loss')
ax.set_title(r'Distortion $\tau=5$')
ax.legend(loc=0, handlelength=1)

tau=50
lr=5.0
train_dir = f'./results/rosenbrock-dim20-batch50/{name}-{depth}-tau{tau}'
gd_mln = scipy.io.loadmat(f'{train_dir}/adam-lr{lr:.2f}.mat')
bs_fwd = scipy.io.loadmat(f'{train_dir}/FWD-PLNet.mat')
bs_dys = scipy.io.loadmat(f'{train_dir}/DYS-PLNet.mat')

ax = axes[1]
ax.semilogy(bs_dys['step'][0, :], bs_dys['vgap'][0, :], linewidth=2, label=r'DYS$\left(\frac{\mu}{\nu-\mu}\right)$')
ax.semilogy(bs_fwd['step'][0, :], bs_fwd['vgap'][0, :], linewidth=2, label=r'FSM$\left(\frac{\mu}{\nu^2}\right)$')
ax.semilogy(gd_mln['step'][0, :], gd_mln['vgap'][0, :], linewidth=2, label='ADAM(5.0)')
ax.legend(loc=1, handlelength=1)
ax.set_xlabel('Steps')
ax.set_xlim(0, 100)
ax.set_title(r'Distortion $\tau=50$')
# ax.grid()

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig(f'./results/rosenbrock-dim20-batch50/rosenbrock20-solver.pdf')

