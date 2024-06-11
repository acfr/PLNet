import os 
import jax 
import jax.numpy as jnp
import jax.random as random
from typing import Sequence, Union, Callable
import optax
import scipy.io 
from flax.training import train_state, orbax_utils
from layer import *
from icnn import ICNN
import orbax.checkpoint
import numpy as np

ROSENBROCK = jax.vmap(lambda x, y, a, b: (x-a) ** 2 / 200. + 0.5 * (y - b * x ** 2) ** 2) 
SINE = jax.vmap(lambda x, y: 0.25*(jnp.sin(8*(x-1.0)-jnp.pi/2) + jnp.sin(8*(y-1.0)-jnp.pi/2)+2.0)) 

def Sampler(
    rng: random.PRNGKey, 
    batch: int,
    xy_dim: int = 2,
    ab_dim: int = 2,
    xy_max: float = 2.,
    ab_max: float = 1.,
):
    rng_xy, rng_ab = random.split(rng)
    xy = random.uniform(rng_xy, (batch, xy_dim), minval=-xy_max, maxval=xy_max) + jnp.array([0., 1.])
    ab = random.uniform(rng_ab, (batch, ab_dim), minval=-ab_max, maxval=ab_max)
    return xy, ab 

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

def data_gen(
    rng: random.PRNGKey,
    has_sine: bool = False,
    train_batch_size: int = 200,
    test_batch_size: int = 5000,
    train_batches: int = 50,
    test_batches: int = 1 
):
    train_size, test_size = train_batch_size * train_batches, test_batch_size * test_batches
    rng_train, rng_test = random.split(rng)
    
    xtrain, ptrain = Sampler(rng_train, train_size)
    xtest, ptest = Sampler(rng_test, test_size)
    if has_sine:
        ft = lambda xy, ab: ROSENBROCK(xy[..., 0], xy[..., 1], ab[..., 0], ab[..., 1]) + SINE(xy[..., 0], xy[..., 1])
    else: 
        ft = lambda xy, ab: ROSENBROCK(xy[..., 0], xy[..., 1], ab[..., 0], ab[..., 1])

    ytrain, ytest = ft(xtrain, ptrain), ft(xtest, ptest)
    
    data = {
        "xtrain": xtrain, 
        "ptrain": ptrain,
        "ytrain": ytrain, 
        "xtest": xtest, 
        "ptest": ptest,
        "ytest": ytest, 
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "x_dim": 2,
        "p_dim": 2
    }

    return data

def train(
    rng,
    model,
    data,
    name: str = 'plnet',
    train_dir: str = './results/rosenbrock',
    lr_max: float = 1e-3,
    epochs: int = 200
):
    ckpt_dir = f'{train_dir}/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    x_dim, p_dim = data['x_dim'], data['p_dim']
    train_batches = data['train_batches']
    train_batch_size = data['train_batch_size']
    idx_shp = (train_batches, train_batch_size)
    train_size = train_batches * train_batch_size

    rng, rng_model = random.split(rng)
    params = model.init(rng_model, jnp.ones(x_dim), jnp.ones(p_dim))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'model: {name}, size: {param_count/1000:.2f}K')

    total_steps = train_batches * epochs
    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=lr_max,
                                           pct_start=0.2, 
                                           pct_final=0.5,
                                           div_factor=10., 
                                           final_div_factor=400.)
    opt = optax.adam(learning_rate=scheduler)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=opt)
    
    @jax.jit
    def fitloss(state, params, x, p, y):
        yh = state.apply_fn(params, x, p)
        loss = optax.l2_loss(yh, y).mean()
        return loss
    
    @jax.jit
    def train_step(state, x, p, y):
        grad_fn = jax.value_and_grad(fitloss, argnums=1)
        loss, grads = grad_fn(state, state.params, x, p, y)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(train_batches):
            x = data['xtrain'][idx[b, :], :] 
            p = data['ptrain'][idx[b, :], :]
            y = data['ytrain'][idx[b, :]]
            model_state, loss = train_step(model_state, x, p, y)
            tloss += loss
        tloss /= train_batches
        train_loss.append(tloss)

        vloss = fitloss(model_state, model_state.params, data['xtest'], data['ptest'], data['ytest'])
        val_loss.append(vloss)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss:.4f}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)

x_dim = 2
mu, nu = 0.04, 16.0
epochs = 600
has_sine = False
rng = random.PRNGKey(42)
rng, rng_data = random.split(rng, 2)
data = data_gen(rng_data, has_sine=has_sine, train_batches=50)

root_dir = './results/rosenbrock-p'
depth = 2
name = 'PBiLipNet'
units = [128]*4 
po_units = [64, 128, x_dim]
pb_units = [64, 128, 256, sum(units)]
block = PBiLipNet(units=units, po_units=po_units, pb_units=pb_units, depth=depth, mu=mu, nu=nu)
model = PPLNet(block)
train_dir = f'{root_dir}/{name}-mu{mu:.2f}-nu{nu:.1f}'
train(rng, model, data, name=name, train_dir=train_dir, epochs=epochs)

