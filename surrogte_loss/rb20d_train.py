import os 
import jax 
import jax.numpy as jnp
import jax.random as random
from typing import Sequence, Union, Callable
import optax
import scipy.io 
from flax.training import train_state, orbax_utils
from layer import cayley, Unitary, QuadPotential, LipNonlin, LipSwish
import orbax.checkpoint
from rosenbrock_utils import *
from flax import linen as nn 
import numpy as np

# different from layer.py, this version allows model to tune its Lipschitz bound nu

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

class iResNet(nn.Module):
    units: Sequence[int]
    depth: int
    tau: jnp.float32 = 10.0
    act_fn: Callable = nn.relu

    def get_bounds(self):
        lognu = self.variables['params']['lognu']
        nu = jnp.squeeze(jnp.exp(lognu), 0)
        mu = nu / self.tau

        return mu, nu, self.tau

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        lognu = self.param('lognu', nn.initializers.constant(jnp.log(4.0)), (1,), jnp.float32)
        nu = jnp.exp(lognu)
        mu = nu / self.tau 
        m = (mu) ** (1. / self.depth)
        n = (nu) ** (1. / self.depth)
        a = 0.5 * (m + n)
        g = (n - m) / (n + m)
        for _ in range(self.depth):
            x = a * x
            x = x + LipNonlin(self.units, gamma=g, act_fn=self.act_fn)(x)

        return x 

class iDenseNet(nn.Module):
    units: Sequence[int]
    depth: int 
    tau: jnp.float32
    use_lipswich: bool = False
    

    def setup(self):
        if self.use_lipswich:
            self.act_fn = LipSwish()
        else:
            self.act_fn = nn.relu

    def get_bounds(self):
        lognu = self.variables['params']['lognu']
        nu = jnp.squeeze(jnp.exp(lognu), 0)
        mu = nu / self.tau

        return mu, nu, self.tau
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        lognu = self.param('lognu', nn.initializers.constant(jnp.log(4.0)), (1,), jnp.float32)
        nu = jnp.exp(lognu)
        mu = nu / self.tau 
        m = (mu) ** (1. / self.depth)
        n = (nu) ** (1. / self.depth)
        a = 0.5 * (m + n)
        g = (n - m) / (n + m)
        for _ in range(self.depth):
            x = a * x 
            x = x + LipNonlin(self.units, gamma=g, act_fn=self.act_fn)(x)

        return x 
    
class PLNet(nn.Module):
    BiLipBlock: nn.Module

    def gmap(self, x: jnp.array) -> jnp.array:
        return self.BiLipBlock(x)

    def get_bounds(self):
        return self.BiLipBlock.get_bounds()
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        x = self.BiLipBlock(x)
        y = QuadPotential()(x)

        return y 
        
def data_gen(
    rng: random.PRNGKey,
    data_dim: int = 20, 
    val_min: float = -2.,
    val_max: float = 2.,
    train_batch_size: int = 200,
    test_batch_size: int = 5000,
    train_batches: int = 200,
    test_batches: int = 1,
    eval_batch_size: int = 5000,
    eval_batches: int = 100,
):
    rng_train, rng_test, rng_eval = random.split(rng, 3)
    
    xtrain = Sampler(rng_train, 
                     train_batch_size * train_batches,
                     data_dim, 
                     x_min=val_min, x_max=val_max)
    xtest  = Sampler(rng_test, 
                     test_batch_size * test_batches,
                     data_dim, 
                     x_min=val_min, x_max=val_max)
    xeval  = Sampler(rng_eval, 
                     eval_batch_size * eval_batches, 
                     data_dim, 
                     x_min=val_min, x_max=val_max)

    ytrain, ytest, yeval = Rosenbrock(xtrain), Rosenbrock(xtest), Rosenbrock(xeval)
    
    data = {
        "xtrain": xtrain, 
        "ytrain": ytrain, 
        "xtest": xtest, 
        "ytest": ytest, 
        "xeval": xeval,
        "yeval": yeval,
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "test_batches": test_batches,
        "test_batch_size": test_batch_size,
        "eval_batches": eval_batches,
        "eval_batch_size": eval_batch_size,
        "data_dim": data_dim
    }

    return data

def train(
    rng,
    model,
    data,
    name: str = 'bilipnet',
    train_dir: str = './results/rosenbrock-nd',
    lr_max: float = 1e-3,
    epochs: int = 600
):

    ckpt_dir = f'{train_dir}/ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dim = data['data_dim']
    train_batches = data['train_batches']
    train_batch_size = data['train_batch_size']

    idx_shp = (train_batches, train_batch_size)
    train_size = train_batches * train_batch_size

    rng, rng_model = random.split(rng)
    params = model.init(rng_model, jnp.ones(data_dim))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'model: {name}, size: {param_count/1000000:.2f}M')

    total_steps = train_batches * epochs
    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=lr_max,
                                           pct_start=0.25, 
                                           pct_final=0.7,
                                           div_factor=10., 
                                           final_div_factor=200.)
    opt = optax.adam(learning_rate=scheduler)
    model_state = train_state.TrainState.create(apply_fn=model.apply,
                                                params=params,
                                                tx=opt)
    
    @jax.jit
    def fitloss(state, params, x, y):
        yh = state.apply_fn(params, x)
        loss = optax.l2_loss(yh, y).mean()
        return loss
    
    @jax.jit
    def train_step(state, x, y):
        grad_fn = jax.value_and_grad(fitloss, argnums=1)
        loss, grads = grad_fn(state, state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss 
    
    train_loss, val_loss = [], []
    Lipmin, Lipmax, Tau = [], [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(train_batches):
            x = data['xtrain'][idx[b, :], :] 
            y = data['ytrain'][idx[b, :]]
            model_state, loss = train_step(model_state, x, y)
            tloss += loss
        tloss /= train_batches
        train_loss.append(tloss)

        vloss = fitloss(model_state, model_state.params, data['xtest'], data['ytest'])
        val_loss.append(vloss)

        lipmin, lipmax, tau = model.apply(model_state.params, method=model.get_bounds)
        Lipmin.append(lipmin)
        Lipmax.append(lipmax)
        Tau.append(tau)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss:.4f}, tau: {tau:.1f}, Lip: {lipmin:.3f}/{lipmax:.2f}')

    eloss = fitloss(model_state, model_state.params, data['xeval'], data['yeval'])
    print(f'{name}: eval loss: {eloss:.4f}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['lipmin'] = jnp.array(Lipmin)
    data['lipmax'] = jnp.array(Lipmax)
    data['tau'] = jnp.array(Tau)
    data['eval_loss'] = eloss

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)

data_dim = 20
lr_max = 1e-2
epochs = 800
n_batch = 50


root_dir = f'./results/rosenbrock-dim{data_dim}-batch{n_batch}'
rng = random.PRNGKey(42)
rng, rng_data = random.split(rng, 2)
data= data_gen(rng_data, train_batches=n_batch, data_dim=data_dim)

name = 'BiLipNet'
depth = 2 
for tau in [2, 4, 5, 8, 10, 20, 50, 70, 100]:
    train_dir = f'{root_dir}/{name}-{depth}-tau{tau}'
    block = BiLipNet([256]*8, depth=depth, tau=tau)
    model = PLNet(block)
    train(rng, model, data, name=name, train_dir=train_dir, lr_max=lr_max, epochs=epochs)


root_dir = f'./results/rosenbrock-dim{data_dim}-batch{n_batch}'
rng = random.PRNGKey(42)
rng, rng_data = random.split(rng, 2)
data= data_gen(rng_data, train_batches=n_batch, data_dim=data_dim)

name = 'i-ResNet'
depth = 5 
for tau in [2., 5., 10., 40., 60., 80., 100.]:
    train_dir = f'{root_dir}/{name}-{depth}-tau{tau:.0f}'
    block = iResNet([640]*2, depth=depth, tau=tau)
    model = PLNet(block)
    train(rng, model, data, name=name, train_dir=train_dir, lr_max=lr_max, epochs=epochs)


root_dir = f'./results/rosenbrock-dim{data_dim}-batch{n_batch}'
rng = random.PRNGKey(42)
rng, rng_data = random.split(rng, 2)
data= data_gen(rng_data, train_batches=n_batch, data_dim=data_dim)

name = 'i-DenseNet'
depth = 5 
for tau in [2., 5., 10., 40., 60., 80., 100.]:
    train_dir = f'{root_dir}/{name}-{depth}-tau{tau:.0f}'
    block = iDenseNet([560]*4, depth=depth, tau=tau)
    model = PLNet(block)
    train(rng, model, data, name=name, train_dir=train_dir, lr_max=lr_max, epochs=epochs)
