# This file is modified from the following files and remains under the
# original licensing.
#
# https://github.com/ott-jax/ott/blob/main/ott/core/icnn.py
# and
# https://github.com/ott-jax/ott/blob/main/ott/core/layers.py
#
#
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.random import PRNGKey

from flax import linen as nn

from typing import Any, Sequence, Callable
ModuleDef = Any

batch_dot = jax.vmap(jnp.dot)

class ActNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        n = x.shape[-1]

        def init_log_scale(dtype=jnp.float_):
            def init(key, shape, dtype=dtype):
                assert x.ndim == 2
                dtype = dtypes.canonicalize_dtype(dtype)
                return jnp.log(x.std(axis=0))
            return init

        def init_bias(dtype=jnp.float_):
            def init(key, shape, dtype=dtype):
                assert x.ndim == 2
                dtype = dtypes.canonicalize_dtype(dtype)
                return -x.mean(axis=0)
            return init

        log_scale = self.param('log_scale', init_log_scale(), [n])
        bias = self.param('bias', init_bias(), [n])
        x = (x + bias) / jnp.exp(log_scale)
        return x

class PositiveDense(nn.Module):
    dim_hidden: int
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Any = None

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param(
            'kernel', self.kernel_init, (inputs.shape[-1], self.dim_hidden))
        kernel = jnp.asarray(kernel, self.dtype)
        kernel = nn.softplus(kernel)
        y = jax.lax.dot_general(
            inputs, kernel, (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision)

        gain = 1./inputs.shape[-1]
        y *= gain
        return y


class ICNN(nn.Module):
    dim_hidden: Sequence[int]
    act_fn: Callable = nn.elu
    actnorm: bool = False  # causing NaN

    def setup(self):
        kernel_init = nn.initializers.variance_scaling(
            1., "fan_in", "truncated_normal")

        num_hidden = len(self.dim_hidden)

        w_zs = list()
        for i in range(1, num_hidden):
            w_zs.append(PositiveDense(self.dim_hidden[i], kernel_init=kernel_init))
        w_zs.append(PositiveDense(1, kernel_init=kernel_init))
        self.w_zs = w_zs

        w_xs = list()
        for i in range(num_hidden):
            w_xs.append(nn.Dense(
                self.dim_hidden[i], use_bias=True,
                kernel_init=kernel_init))

        w_xs.append(nn.Dense(1, use_bias=True, kernel_init=kernel_init))
        self.w_xs = w_xs


    @nn.compact
    def __call__(self, x):
        single = x.ndim == 1
        if single:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2

        z = self.act_fn(self.w_xs[0](x))
        for Wz, Wx in zip(self.w_zs[:-1], self.w_xs[1:-1]):
            z = Wz(z) + Wx(x)
            if self.actnorm:
                z = ActNorm()(z)
            z = self.act_fn(z)

        # An activation on this last layer is really helpful sometimes.
        y = self.act_fn(self.w_zs[-1](z) + self.w_xs[-1](x))
        y = jnp.squeeze(y, -1)

        log_alpha = self.param(
            'log_alpha', nn.initializers.constant(0), [])
        y += jnp.exp(log_alpha)*0.5*batch_dot(x, x)

        # if single:
        #     y = jnp.squeeze(y, 0)
        return y
