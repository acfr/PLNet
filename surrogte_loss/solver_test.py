import os
import jax 
import jax.numpy as jnp
import jax.random as random 
from flax import linen as nn 
from layer import MonLipNet
import scipy.io 

# default activation: relu
# it satisfies the property relu(x)=\prox_f^{\alpha}(x) for any \alpha > 0
# for other activation, one need to define \prox_f^{\alpha}, although \sigma(x)=\prox_f^1(x).
# see the list in appendix

# params = MonLipNet.get_params()
# params = {
#     "mu": float,
#     "gam": float,
#     "units": Sequence[int],
#     "V": Sequence[jnp.arry]
#     "S": jnp.array,
#     "by": jnp.array,
#     "bh": jnp.array
# }

def mln_fwd(params, x):
    b = mln_fwd_x2b(params, x)
    z = mln_fwd_b2z(params, b)
    y = mln_fwd_xz2y(params, x, z)

    return y, z 

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

def mln_bwd_err(params, bz, z):
    zv = mln_bwd_z2v(params, z)
    zh = mln_fwd_b2z(params, bz-zv)
    ze = zh - z 
    return jnp.linalg.norm(ze, axis=-1)

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

# Eq. (13)
def ForwardMethod(params, x, y, z0, alpha,
    max_iter: int = 2000,
    epsilon: float = 1e-3
):
    xerr, zerr = [], []
    bz = mln_bwd_y2b(params, y)
    xk = mln_bwd_yz2x(params, y, z0)
    for k in range(max_iter):
        yk, zk = mln_fwd(params, xk)
        xk += alpha*(y - yk)
        xe = jnp.linalg.norm(xk-x, axis=-1) / (jnp.linalg.norm(xk) + 1e-3)
        ek = mln_bwd_err(params, bz, zk) / (jnp.linalg.norm(zk) + 1e-3)
        mek = jnp.mean(ek)
        if k % 10 == 0:
            print(f'FWM: {k+1:4d} | err: {mek:.4f}, xerr: {jnp.max(xe):.4f}')
        zerr.append(jnp.mean(ek))
        xerr.append(jnp.mean(xe))
        # z.append(zk)
        if mek <= epsilon:
            print(f'FWM: {k} steps, err: {mek:.4f}, xerr: {jnp.max(xe):.4f}')
            break 

    return jnp.array(xerr), jnp.array(zerr) 
        
# Eq. (15)
def DavisYinSplit(params, x, y, z0, alpha,
    max_iter: int = 2000,
    epsilon: float = 1e-3
):
    bz = mln_bwd_y2b(params, y)
    uk = z0 # jnp.zeros(jnp.shape(bz))
    xerr, zerr = [], []
    for k in range(max_iter):
        zh = nn.relu(uk)
        uh = 2*zh - uk 
        zk = mln_RA(params, alpha, bz, zh, uh)
        uk += zk - zh 
        ek = mln_bwd_err(params, bz, zk) / (jnp.linalg.norm(zk) + 1e-3)
        xk = mln_bwd_yz2x(params, y, zk)
        xe = jnp.linalg.norm(xk-x, axis=-1) / (jnp.linalg.norm(xk) + 1e-3)
        mek = jnp.mean(ek)
        if k % 10 == 0:
            print(f'DYS: {k+1:4d} | zerr: {mek:.4f}, xerr: {jnp.max(xe):.4f}')
        zerr.append(jnp.mean(ek))
        xerr.append(jnp.mean(xe))
        if mek <= epsilon:
            print(f'DYS: {k} steps, err: {mek:.4f}, xerr: {jnp.max(xe):.4f}')
            break 

    return jnp.array(xerr), jnp.array(zerr) 

root_dir = './results/solver'
os.makedirs(root_dir, exist_ok=True)

for hid_dim in [128, 64]:
    for data_dim in [16, 8]:
        rng = random.PRNGKey(42)        
        mu, nu = 0.2, 4.0 
        batch = 32
        units = 6*[hid_dim]
        model = MonLipNet(units, mu, nu)
        p = model.init(rng, jnp.ones(data_dim))
        params = model.apply(p, method=model.get_params)
        rng_x, rng_z = random.split(rng)
        x = random.normal(rng_x, (batch, data_dim))
        z0 = random.normal(rng_z, (batch, sum(units)))
        y = model.apply(p, x)

        data = {
            'mu': mu,
            'nu': nu,
            'params': params,
            'x': x,
            'y': y
        }

        alpha = mu / (nu ** 2)
        xerr, zerr = ForwardMethod(params, x, y, z0, alpha)
        data['xfwm'] = xerr 
        data['zfwm'] = zerr 

        for a in [1.0, 0.7, 0.5, 0.3]:
            alpha = a * mu / nu 
            xerr, zerr = DavisYinSplit(params, x, y, z0, alpha)
            data[f'xdys{a:.1f}'] = xerr
            data[f'zdys{a:.1f}'] = zerr 

        scipy.io.savemat(f'{root_dir}/MonLipNet-nx{data_dim}-nz{sum(units)}-mu{mu:.1f}-nu{nu:.1f}.mat', data)
