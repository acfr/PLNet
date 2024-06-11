import jax 
import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable

def cayley(W):
    m, n = W.shape 
    if n > m:
       return cayley(W.T).T
    
    U, V = W[:n, :], W[n:, :]
    Z = (U - U.T) + (V.T @ V)
    I = jnp.eye(n)
    Zi = jnp.linalg.inv(I+Z)

    return jnp.concatenate([Zi @ (I-Z), -2 * V @ Zi], axis=0)

class Unitary(nn.Module):
    units: int = 0
    use_bias: bool = True 

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        n = jnp.shape(x)[-1]
        m = n if self.units == 0 else self.units
        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (m, n),
                       jnp.float32)
        a = self.param('a', 
                       nn.initializers.constant(jnp.linalg.norm(W)), 
                       (1,),
                       jnp.float32)

        R = cayley((a / jnp.linalg.norm(W)) * W)
        z = x @ R.T 
        if self.use_bias: 
            b = self.param('b', nn.initializers.zeros_init(), (m,), jnp.float32)
            z += b

        return z 
    
    def get_params(self):
        W = self.variables['params']['W']
        a = self.variables['params']['a']
        R = cayley((a / jnp.linalg.norm(W)) * W)
        b = self.variables['params']['b'] if self.use_bias else 0. 

        params = {
            'R': R,
            'b': b
        }

        return params

class MonLipNet(nn.Module):
    units: Sequence[int]
    mu: jnp.float32 = 0.1 # Monotone lower bound
    nu: jnp.float32 = 10.0 # Lipschitz upper bound (nu > mu)
    # act_fn: Callable = nn.relu

    def get_params(self):
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
            "mu": self.mu,
            "gam": self.nu - self.mu,
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
        by = self.param('by', nn.initializers.zeros_init(), (nx,), jnp.float32) 
        y = self.mu * x + by 
        
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (nx, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq) 
        sqrt_2g, sqrt_g2 = jnp.sqrt(2. * (self.nu - self.mu)), jnp.sqrt((self.nu - self.mu) / 2.)
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
    mu: float = 0.1
    nu: float = 10.
    depth: int = 2

    def setup(self):
        uni, mon = [], []
        mu = self.mu ** (1. / self.depth)
        nu = self.nu ** (1. / self.depth)
        for _ in range(self.depth):
            uni.append(Unitary())
            mon.append(MonLipNet(self.units, mu=mu, nu=nu))
        uni.append(Unitary())
        self.uni = uni
        self.mon = mon

    def __call__(self, x: jnp.array) -> jnp.array:
        for k in range(self.depth):
            x = self.uni[k](x)
            x = self.mon[k](x)
        x = self.uni[self.depth](x)
        return x 
        
class MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
        x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x
  
class LipLinear(nn.Module):
    unit: int 
    gamma: float = 1.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        n = jnp.shape(x)[-1]
        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (self.unit, n),
                       jnp.float32)
        b = self.param('b', nn.initializers.zeros_init(), (self.unit,), jnp.float32) if self.use_bias else 0.
        x = self.gamma / jnp.linalg.norm(W) * x @ W.T + b
        return x 

class LipNonlin(nn.Module):
    units: Sequence[int] 
    gamma: float = 1.0
    use_bias: bool = True
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        n = jnp.shape(x)[-1]
        for unit in self.units:
            x = LipLinear(unit, gamma=1., use_bias=self.use_bias)(x)
            x = self.act_fn(x)
        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (n, self.units[-1]),
                       jnp.float32)
        x = self.gamma / jnp.linalg.norm(W) * x @ W.T
        return x 
    
class iResNet(nn.Module):
    units: Sequence[int]
    depth: int
    mu: float
    nu: float
    act_fn: Callable = nn.relu

    def setup(self):
        m = (self.mu) ** (1. / self.depth)
        n = (self.nu) ** (1. / self.depth)
        self.a = 0.5 * (m + n)
        self.g = (n - m) / (n + m)

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        for _ in range(self.depth):
            x = self.a * x
            x = x + LipNonlin(self.units, gamma=self.g, act_fn=self.act_fn)(x)

        return x 

class LipSwish(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        beta = self.param('beta', nn.initializers.constant(0.5), (1,), jnp.float32)
        x = x * nn.sigmoid(beta * x) / 1.1
        return x 
                
class iDenseNet(nn.Module):
    units: Sequence[int]
    depth: int 
    mu: float
    nu: float
    use_lipswich: bool = False
    

    def setup(self):
        m = (self.mu) ** (1. / self.depth)
        n = (self.nu) ** (1. / self.depth)
        self.a = 0.5 * (m + n)
        self.g = (n - m) / (n + m)
        if self.use_lipswich:
            self.act_fn = LipSwish()
        else:
            self.act_fn = nn.relu

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        for _ in range(self.depth):
            x = self.a * x 
            x = x + LipNonlin(self.units, gamma=self.g, act_fn=self.act_fn)(x)

        return x 

class QuadPotential(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        c = self.param('c', nn.initializers.constant(0.), (1,), jnp.float32)
        y = 0.5 * (jnp.linalg.norm(x, axis=-1) ** 2) + c
        return y
    
class PLNet(nn.Module):
    BiLipBlock: nn.Module

    def gmap(self, x: jnp.array) -> jnp.array:
        return self.BiLipBlock(x)

    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        x = self.BiLipBlock(x)
        y = QuadPotential()(x)

        return y 
        
class PUnitary(nn.Module):
    units: int = 0

    @nn.compact
    def __call__(self, x: jnp.array, b: jnp.array) -> jnp.array:
        n = jnp.shape(x)[-1]
        m = n if self.units == 0 else self.units
        W = self.param('W', 
                       nn.initializers.glorot_normal(), 
                       (m, n),
                       jnp.float32)
        a = self.param('a', 
                       nn.initializers.constant(jnp.linalg.norm(W)), 
                       (1,),
                       jnp.float32)

        R = cayley((a / jnp.linalg.norm(W)) * W)
        z = x @ R.T + b

        return z 
    
    def get_params(self):
        W = self.variables['params']['W']
        a = self.variables['params']['a']
        R = cayley((a / jnp.linalg.norm(W)) * W)

        params = {
            'R': R
        }

        return params
    
class PMonLipNet(nn.Module):
    units: Sequence[int]
    mu: jnp.float32 = 0.1 # Monotone lower bound
    nu: jnp.float32 = 10.0 # Lipschitz upper bound (nu > mu)
    # act_fn: Callable = nn.relu

    def get_params(self):
        Fq = self.variables['params']['Fq']
        fq = self.variables['params']['fq']
        Q = cayley((fq / jnp.linalg.norm(Fq)) * Fq).T 
        V, S = [], []
        idx = 0
        L = len(self.units)
        for k, nz in zip(range(L), self.units):
            Qk = Q[idx:idx+nz, :] 
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
            "mu": self.mu,
            "gam": self.nu - self.mu,
            "units": self.units,
            "V": V, 
            "S": S,
            "by": by
        }

        return params

    @nn.compact
    def __call__(self, x: jnp.array, b: jnp.array) -> jnp.array:
        nx = jnp.shape(x)[-1]  
        by = self.param('by', nn.initializers.zeros_init(), (nx,), jnp.float32) 
        y = self.mu * x + by 
        
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (nx, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        QT = cayley((fq / jnp.linalg.norm(Fq)) * Fq) 
        sqrt_2g, sqrt_g2 = jnp.sqrt(2. * (self.nu - self.mu)), jnp.sqrt((self.nu - self.mu) / 2.)
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
            # use relu activation, no need for psi
            # pk = self.param(f'p{k}', nn.initializers.zeros_init(), (nz,), jnp.float32)
            zk = nn.relu(2 * (zk @ Ak_1) @ BTk + sqrt_2g * x @ STk + b[..., idx:idx+nz])
            # zk = nn.relu(zk * jnp.exp(-pk)) * jnp.exp(pk)
            y += sqrt_g2 * zk @ STk.T  
            idx += nz 
            nz_1 = nz 
            Ak_1 = ATk.T     

        return y 

class PBiLipNet(nn.Module):
    units: Sequence[int]
    po_units: Sequence[int]
    pb_units: Sequence[int]
    mu: float = 0.1
    nu: float = 10.
    depth: int = 2

    def setup(self):
        uni, mon = [], []
        uni_b, mon_b = [], []
        mu = self.mu ** (1. / self.depth)
        nu = self.nu ** (1. / self.depth)
        for _ in range(self.depth):
            uni.append(PUnitary())
            uni_b.append(MLP(self.po_units))
            mon.append(PMonLipNet(self.units, mu=mu, nu=nu))
            mon_b.append(MLP(self.pb_units))
        uni.append(PUnitary())
        uni_b.append(MLP(self.po_units))
        self.uni = uni
        self.mon = mon
        self.uni_b = uni_b
        self.mon_b = mon_b

    def __call__(self, x: jnp.array, p: jnp.array) -> jnp.array:
        for k in range(self.depth):
            b = self.uni_b[k](p)
            x = self.uni[k](x, b)
            b = self.mon_b[k](p)
            x = self.mon[k](x, b)
        b = self.uni_b[self.depth](p)
        x = self.uni[self.depth](x, b)
        return x

class PPLNet(nn.Module):
    PBiLipBlock: nn.Module

    def gmap(self, x: jnp.array, p: jnp.array) -> jnp.array:
        return self.BiLipBlock(x, p)

    @nn.compact
    def __call__(self, x: jnp.array, p: jnp.array) -> jnp.array:
        x = self.PBiLipBlock(x, p)
        y = QuadPotential()(x)

        return jnp.squeeze(y) 