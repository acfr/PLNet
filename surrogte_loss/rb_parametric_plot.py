import os 
import jax 
import jax.numpy as jnp
from typing import Sequence, Callable
import scipy.io 
import matplotlib.pyplot as plt
from layer import *
import orbax.checkpoint
import numpy as np

ROSENBROCK = jax.vmap(lambda x, y, a, b: (x-a) ** 2 / 200. + 0.5 * (y - b*x ** 2) ** 2) 
SINE = jax.vmap(lambda x, y: 0.25*(jnp.sin(8*(x-1.0)-jnp.pi/2) + jnp.sin(8*(y-1.0)-jnp.pi/2)+2.0)) 

def MeshField(
    x_range: Sequence[float] = [-2., 2.],
    y_range: Sequence[float] = [-1., 3.],
    n_grid: int = 200
):
    x = jnp.linspace(x_range[0], x_range[1], n_grid)
    y = jnp.linspace(y_range[0], y_range[1], n_grid)
    xx, yy = jnp.meshgrid(x, y)
    z = jnp.concatenate([jnp.reshape(xx,(-1,1)), jnp.reshape(yy, (-1,1))], axis=1)
    return z, xx, yy 

x_dim = 2
mu, nu = 0.04, 16.0
epochs = 600
has_sine = False

root_dir = './results/rosenbrock-p'
depth = 2
name = 'PBiLipNet'
units = [128]*4 
po_units = [64, 128, x_dim]
pb_units = [64, 128, 256, sum(units)]
block = PBiLipNet(units=units, po_units=po_units, pb_units=pb_units, depth=depth, mu=mu, nu=nu)
model = PPLNet(block)
train_dir = f'{root_dir}/{name}-mu{mu:.2f}-nu{nu:.1f}'
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
params = orbax_checkpointer.restore(f'{train_dir}/ckpt/params')

Fn = lambda xy, ab: ROSENBROCK(xy[:,0], xy[:,1], ab[:,0], ab[:,1])
fn = lambda xy, p: model.apply(params, xy, p)

xy, x, y = MeshField()
batches = jnp.shape(xy)[0]
ones = jnp.ones((batches, 1))
x_shp = jnp.shape(x)

a, b = 1.0, 1.0
ab = jnp.concatenate((a*ones, b*ones), axis=-1)
zh1, zt1 = fn(xy, ab), Fn(xy, ab)
zh1 = jnp.reshape(zh1, x_shp)
zt1 = jnp.reshape(zt1, x_shp)

a, b = 0.0, 0.0
ab = jnp.concatenate((a*ones, b*ones), axis=-1)
zh2, zt2 = fn(xy, ab), Fn(xy, ab)
zh2 = jnp.reshape(zh2, x_shp)
zt2 = jnp.reshape(zt2, x_shp)

a, b = -1., -1.
ab = jnp.concatenate((a*ones, b*ones), axis=-1)
zh3, zt3 = fn(xy, ab), Fn(xy, ab)
zh3 = jnp.reshape(zh3, x_shp)
zt3 = jnp.reshape(zt3, x_shp)

dat = {
    "x": x,
    "y": y,
    "zh1": zh1,
    "zt1": zt1,
    "zh2": zh2,
    "zt2": zt2,
    "zh3": zh3,
    "zt3": zt3
}

scipy.io.savemat(f'{train_dir}/plot_dat.mat', dat)

dat = scipy.io.loadmat(f'{train_dir}/plot_dat.mat')
x = jnp.array(dat['x'])
y = jnp.array(dat['y'])
zh1 = jnp.array(dat['zh1'])
zt1 = jnp.array(dat['zt1'])
zh2 = jnp.array(dat['zh2'])
zt2 = jnp.array(dat['zt2'])
zh3 = jnp.array(dat['zh3'])
zt3 = jnp.array(dat['zt3'])

zmax = jnp.max(zt3)
emax = jnp.max(jnp.abs(zh3-zt3))

lvl, lvl2 = 30, 10
plt.rcParams['font.size'] = 14
plt.rcParams['text.usetex'] = True

fig, axes = plt.subplots(2, 3, figsize=(9, 5))

ax = axes[0, 0]
cs = ax.contourf(x, y, zt1, levels=lvl, cmap="RdYlGn_r")
cbar = fig.colorbar(cs, ax=ax)
ax.set_ylabel('True')
ax.set_title(r'$(a,b)=(1,1)$')
ax.set_xticks([])
ax.set_yticks([])

ax = axes[1, 0]
cs = ax.contourf(x, y, zh1, levels=lvl, cmap="RdYlGn_r")
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(cs, ax=ax)
ax.set_ylabel('Partially BiLipNet')

ax = axes[0, 1]
cs = ax.contourf(x, y, zt2, levels=lvl, cmap="RdYlGn_r")
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(cs, ax=ax)
ax.set_title(r'$(a,b)=(0,0)$')

ax = axes[1, 1]
cs = ax.contourf(x, y, zh2, levels=lvl, cmap="RdYlGn_r")
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(cs, ax=ax)

ax = axes[0, 2]
cs = ax.contourf(x, y, zt3, levels=lvl, cmap="RdYlGn_r")
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(cs, ax=ax)
ax.set_title(r'$(a,b)=(-1,-1)$')
ax = axes[1, 2]
cs = ax.contourf(x, y, zh3, levels=lvl, cmap="RdYlGn_r")
ax.set_xticks([])
ax.set_yticks([])
cbar = fig.colorbar(cs, ax=ax)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
# 
fig.savefig(f'{train_dir}/rosenbrock-parametric.pdf')
plt.close(fig)