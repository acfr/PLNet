import os 
import jax 
import jax.numpy as jnp
import jax.random as random
from typing import Sequence, Union, Callable
import optax
import scipy.io 
import matplotlib.pyplot as plt
from flax.training import train_state, orbax_utils
from layer import *
from icnn import ICNN
import orbax.checkpoint

ROSENBROCK = jax.vmap(lambda x, y: (x-1.0) ** 2 / 200. + 0.5 * (y - x ** 2) ** 2) 
SINE = jax.vmap(lambda x, y: 0.25*(jnp.sin(8*(x-1.0)-jnp.pi/2) + jnp.sin(8*(y-1.0)-jnp.pi/2)+2.0)) 

def xy_sampler(
        rng: random.PRNGKey, 
        shp: Sequence[int], 
        xy_min: Union[float, jnp.ndarray] = -2., 
        xy_max: Union[float, jnp.ndarray] = 2.
):
    return random.uniform(rng, shp, minval=xy_min, maxval=xy_max) + jnp.array([0., 1.])

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
    train_batch_size: int = 100,
    test_batch_size: int = 500,
    train_batches: int = 10,
    test_batches: int = 1 
):
    train_size, test_size = train_batch_size * train_batches, test_batch_size * test_batches
    data_dim = 2
    rng_train, rng_test = random.split(rng, 2)

    corners = jnp.array([[2.0, 3.0], [2.0, -1.0], [-2.0, 3.0], [-2.0, -1.0]])
    
    xtrain = xy_sampler(rng_train, (train_size-4, data_dim))
    xtrain = jnp.concatenate([xtrain, corners])
    xtest = xy_sampler(rng_test, (test_size, data_dim))
    if has_sine:
        ft = lambda xy: ROSENBROCK(xy[..., 0], xy[..., 1]) + SINE(xy[..., 0], xy[..., 1])
    else: 
        ft = lambda xy: ROSENBROCK(xy[..., 0], xy[..., 1])

    ytrain, ytest = ft(xtrain), ft(xtest)
    
    data = {
        "xtrain": xtrain, 
        "ytrain": ytrain, 
        "xtest": xtest, 
        "ytest": ytest, 
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "data_dim": data_dim
    }

    return data, ft  

def train(
    rng,
    model,
    data,
    ft: Callable = None, 
    name: str = 'plnet',
    train_dir: str = './results/rosenbrock',
    lr_max: float = 1e-3,
    epochs: int = 200
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
    LipMin, LipMax = [], []
    for epoch in range(epochs):
        rng, rng_idx = random.split(rng)
        idx = random.permutation(rng_idx, train_size)
        idx = jnp.reshape(idx, idx_shp)
        tloss = 0. 
        for b in range(data['train_batches']):
            x = data['xtrain'][idx[b, :], :] 
            y = data['ytrain'][idx[b, :]]
            model_state, loss = train_step(model_state, x, y)
            tloss += loss
        tloss /= train_batch_size
        train_loss.append(tloss)

        if name == 'ICNN':
            lipmin, lipmax = -1., -1.
        else:
            f = jax.jit(lambda x: model_state.apply_fn(model_state.params, x, method=model.gmap))
            points = jnp.array([[2.0, 3.0], [2.0, -1.0], [-2.0, 3.0], [-2.0, -1.0], [0., 0.], [1., 1.]])
            nsample = 2000
            rng, rng_xl, rng_dx = random.split(rng, 3)
            xl = xy_sampler(rng_xl, (nsample-6, data_dim))
            xl = jnp.concatenate([xl, points])
            dx = random.normal(rng_dx, (nsample, data_dim))
            lipmin, lipmax = estimate_lipschitz(f, xl, dx)

        vloss = fitloss(model_state, model_state.params, data['xtest'], data['ytest'])
        val_loss.append(vloss)
        LipMin.append(lipmin)
        LipMax.append(lipmax)

        print(f'Epoch: {epoch+1:3d} | loss: {tloss:.4f}/{vloss:.4f}, lip: {lipmin:.2f}/{lipmax:.1f}')

    data['train_loss'] = jnp.array(train_loss)
    data['val_loss'] = jnp.array(val_loss)
    data['LipMin'] = jnp.array(LipMin)
    data['LipMax'] = jnp.array(LipMax)
    data['model'] = model_state.params

    fh = jax.jit(lambda x : model_state.apply_fn(model_state.params, x))
    dat = plot_gen(fh, ft, train_dir)
    data['dat'] = dat

    scipy.io.savemat(f'{train_dir}/data.mat', data)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(model_state.params)
    orbax_checkpointer.save(f'{ckpt_dir}/params', model_state.params, save_args=save_args)

def estimate_lipschitz(f, x, dx, eps=0.05):
    norms = lambda x: jnp.linalg.norm(x, axis=-1)
    dy = f(x + eps * dx) - f(x - eps * dx)
    lip = 0.5 * norms(dy) / (eps * norms(dx))
    lipmin = jnp.min(lip)
    lipmax = jnp.max(lip)

    return lipmin, lipmax

def plot_gen(
    fh: Callable,
    ft: Callable,
    train_dir: str 
):
    xy, x, y = MeshField()
    zh, zt = fh(xy), ft(xy)
    zmin, zmax = jnp.min(zt), jnp.max(zt)
    x_shp = jnp.shape(x)
    zh = jnp.reshape(zh, x_shp)
    zt = jnp.reshape(zt, x_shp)
    ze = jnp.abs(zt - zh)

    dat = {
        "x": x,
        "y": y,
        "zh": zh,
        "zt": zt,
        "zmin": zmin,
        "zmax": zmax
    }

    plt.rcParams['figure.figsize'] = (24,8)
    fig,ax = plt.subplots(1,3)
    cs1 = ax[0].contourf(x, y, zt, levels=50, cmap="RdYlGn_r", vmin=zmin, vmax=zmax)
    ax[0].set_title('True',fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    cbar = fig.colorbar(cs1, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)

    cs2 = ax[1].contourf(x, y, zh, levels=50, cmap="RdYlGn_r", vmin=zmin, vmax=zmax)
    ax[1].set_title('fit',fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    cbar = fig.colorbar(cs2, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)

    cs2 = ax[2].contourf(x, y, ze, levels=20)
    ax[2].set_title('error',fontsize=20)
    ax[2].tick_params(axis='both', which='major', labelsize=15)
    cbar = fig.colorbar(cs2, ax=ax[2])
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()

    fig.savefig(f'{train_dir}/fit.pdf', bbox_inches='tight')
    plt.close(fig)

    return dat 

mu, nu = 0.5, 2.
epochs = 600

for has_sine in [True, False]:
    
    if has_sine:
        root_dir = f'./results/rosenbrock-sine'
    else:
        root_dir = './results/rosenbrock'

    for name in ["BiLipNet"]:
        rng = random.PRNGKey(42)
        rng, rng_data = random.split(rng, 2)
        data, ft = data_gen(rng_data, has_sine=has_sine, train_batches=50)

        if name == 'BiLipNet':
            depth = 2
            block = BiLipNet([128]*4, depth=depth, mu=mu, nu=nu)
            model = PLNet(block)
            train_dir = f'{root_dir}/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}'
        elif name == 'ICNN':
            depth = 8
            model = ICNN([180]*8)
            train_dir = f'{root_dir}/{name}-{depth}'
        elif name == 'MLP':
            depth = 3
            block = MLP([128, 256, 256, 512])
            model = PLNet(block)
            train_dir = f'{root_dir}/{name}-{depth}'

        os.makedirs(train_dir, exist_ok=True)
        train(rng, model, data, ft, name=f'{name}', train_dir=train_dir, epochs=epochs)
        
    for name in ["i-ResNet"]:
        for depth in [2, 3, 4, 5, 6, 7, 8]:
            rng = random.PRNGKey(42)
            rng, rng_data = random.split(rng, 2)
            data, ft = data_gen(rng_data, has_sine=has_sine, train_batches=50)
            
            if name == 'i-ResNet':
                widths = [340, 275, 240, 215, 195, 180, 170]
                block = iResNet([widths[depth-2]]*2, depth=depth, mu=mu, nu=nu)
                model = PLNet(block)
            elif name == 'i-DenseNet':
                widths = [196, 160, 140, 125, 112, 105, 98]
                block = iDenseNet([widths[depth-2]]*4, depth=depth, mu=mu, nu=nu, use_lipswich=False)
                model = PLNet(block)

            train_dir = f'{root_dir}/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}'
            os.makedirs(train_dir, exist_ok=True)

            train(rng, model, data, ft, name=f'{name}-{depth}', train_dir=train_dir, epochs=epochs)

