import os 
import jax
import jax.numpy as jnp
import jax.random as random 
import optax
from flax import linen as nn 
from flax.training import train_state
import numpy as np
import scipy.io 
from layer import *

def ft(x:jnp.array) -> jnp.array:
    return 2. * ( x >= 0.) - 2. * (x < 0.)

## calculate the optimal fit for a step function
def fit_opt(xlim, mu, nu):
    xa = (4 - mu * xlim) / (2 * nu - mu) 

    def fopt(x):
        if x <= -xa:
            y = mu * (x + xa) - nu * xa 
        elif x <= xa:
            y = nu * x 
        else:
            y = mu * (x - xa) + nu * xa 
        
        return y 

    return xa, fopt 

def data_gen(rng, xlim=2.0, train_size=1000, test_size=500):
    data_dim = 1 
    xv = jnp.linspace(-xlim, xlim, num=test_size)
    xv = jnp.reshape(xv, (test_size, data_dim))
    yv = ft(xv)

    xt = random.uniform(rng, (train_size, data_dim), minval=-xlim, maxval=xlim)
    yt = ft(xt)

    data = {'xtrain': xt, 'ytrain': yt, 'xtest': xv, 'ytest': yv}

    return data, data_dim

def train(
    rng,
    model, 
    data,
    xa,
    fopt,
    Lr: float = 0.01,
    data_dim: int = 1, 
    train_batch_size: int = 100,
    Epochs: int = 200,
    train_dir: str = './results/toy1d'
):
    n_train = jnp.shape(data['xtrain'])[0]
    n_test = jnp.shape(data['xtest'])[0]
    n_batch = n_train // train_batch_size
    total_steps = Epochs * n_batch + Epochs


    rng, rng_model = random.split(rng)
    params = model.init(rng_model, jnp.ones(data_dim))
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f'{mdl_name} size: {param_count/1000:.2f}K')

    xo = np.array(data['xtest'])
    yo = np.zeros_like(xo)
    for k in range(n_test):
        yo[k] = fopt(xo[k])
    yopt = jnp.asarray(yo)
    loss_opt = optax.l2_loss(data['ytest'], yopt).mean()

    scheduler = optax.linear_onecycle_schedule(transition_steps=total_steps, 
                                           peak_value=Lr,
                                           pct_start=0.25, 
                                           pct_final=0.7,
                                           div_factor=10., 
                                           final_div_factor=100.)

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
    Lipmin, Lipmax = [], []
    for epoch in range(Epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, n_train)
        idx = jnp.reshape(idx, (n_batch, train_batch_size))
        tloss = 0.
        for b in range(n_batch):
            x = data['xtrain'][idx[b, :], :] 
            y = data['ytrain'][idx[b, :], :]
            model_state, loss = train_step(model_state, x, y)
            tloss += loss
        tloss /= n_batch
        f = jax.jit(lambda x : model_state.apply_fn(model_state.params, x))
        yh = f(data['xtest'])
        vloss = optax.l2_loss(yh, data['ytest']).mean()
        rng, rng_dx = random.split(rng, 2)
        dx = random.uniform(rng_dx, (n_test, data_dim))
        lipmin, lipmax = estimate_lipschitz(f, 2. * data['xtest'], dx)
        train_loss.append(tloss)
        val_loss.append(vloss)
        Lipmin.append(lipmin)
        Lipmax.append(lipmax)
        print(f'{epoch+1:3d} | loss: {tloss:.3f}/{vloss:.3f}/{loss_opt:.3f}, Lip: {lipmin:.2f}/{lipmax:.2f}')

    data['xa'] = xa
    data['yopt'] = yopt
    data['lopt'] = loss_opt
    data['yh'] = yh 
    data['params'] = model_state.params
    data['num_param'] = param_count
    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['lipmin'] = jnp.array(Lipmin)
    data['lipmax'] = jnp.array(Lipmax) 

    scipy.io.savemat(f'{train_dir}/data.mat', data) 

def estimate_lipschitz(f, x, dx, eps=0.05):
    norms = lambda x: jnp.linalg.norm(x, axis=-1)
    dy = f(x + eps * dx) - f(x - eps * dx)
    lip = 0.5 * norms(dy) / (eps * norms(dx))
    lipmin = jnp.min(lip)
    lipmax = jnp.max(lip)

    return lipmin, lipmax
    
root_dir = './results/toy1d'

data_dim = 1
xlim = 2.
mu, nu = 0.1, 10.
xa, loss_opt = fit_opt(xlim, mu, nu)

train_batch_size = 100
test_batch_size = 400
Epochs = 200
Lr = 0.01

for name in ["MonLipNet"]:
    rng = random.PRNGKey(42)
    rng, data_rng = random.split(rng, 2)
    data, data_dim = data_gen(data_rng, xlim=xlim)
    depth = 1
    model = MonLipNet(units=[32] * 8, mu=mu, nu=nu)
    mdl_name = f'{name}-dp{depth}'
    train_dir = f'{root_dir}/{mdl_name}-mu{mu:.1f}-nu{nu:.1f}'
    os.makedirs(train_dir, exist_ok=True)
    train(rng, model, data, 
            xa, loss_opt, Lr=Lr, 
            data_dim=data_dim, 
            Epochs=Epochs, 
            name=mdl_name,
            train_dir=train_dir)

for depth, width in zip([3, 5, 7, 9], [42, 32, 28, 24]):
    mdl_name = f'i-Densenet-dp{depth}'
    rng = random.PRNGKey(42)
    rng, data_rng = random.split(rng, 2)
    data, data_dim = data_gen(data_rng, xlim=xlim)
    model = iDenseNet(units=[width] * 4, depth=depth, mu=mu, nu=nu, use_lipswich=False)
    train_dir = f'{root_dir}/{mdl_name}-mu{mu:.1f}-nu{nu:.1f}'
    os.makedirs(train_dir, exist_ok=True)
    train(rng, model, data, 
        xa, loss_opt, Lr=Lr, 
        data_dim=data_dim, 
        Epochs=Epochs, 
        name=mdl_name,
        train_dir=train_dir)

for depth, width in zip([3, 5, 7, 9], [72, 55, 46, 41]):
    mdl_name = f'i-Resnet-dp{depth}'
    rng = random.PRNGKey(42)
    rng, data_rng = random.split(rng, 2)
    data, data_dim = data_gen(data_rng, xlim=xlim)
    model = iResNet(units=[width] * 2, depth=depth, mu=mu, nu=nu)
    train_dir = f'{root_dir}/{mdl_name}-mu{mu:.1f}-nu{nu:.1f}'
    os.makedirs(train_dir, exist_ok=True)
    train(rng, model, data, 
        xa, loss_opt, Lr=Lr, 
        data_dim=data_dim, 
        Epochs=Epochs, 
        name=mdl_name,
        train_dir=train_dir)