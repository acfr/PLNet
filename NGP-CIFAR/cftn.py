import tensorflow as tf
from tensorflow.python import keras 
import numpy as np 
import edward as ed
import math 
import functools
from absl import logging

import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__

# def cayley(W):
#     _, m, n = W.shape 
#     if m >= n:
#         I = tf.eye(n, dtype=W.dtype)[None,:,:]
#         U, V = W[:, :n, :], W[:, n:, :]
#         Uc = tf.transpose(U, perm=[0,2,1], conjugate=True)
#         Vc = tf.transpose(V, perm=[0,2,1], conjugate=True)
#         Z = U - Uc + Vc @ V 
#         Zi = tf.linalg.inv(I+Z) #tf.linalg.solve(I+Z, I)
#         R = tf.concat([(I-Z) @ Zi, -2 * V @ Zi], axis=1)
#     else:
#         I = tf.eye(m, dtype=W.dtype)[None,:,:]
#         Uc, Vc = W[:, :, :m], W[:, :, m:]
#         U = tf.transpose(Uc, perm=[0,2,1], conjugate=True)
#         V = tf.transpose(Vc, perm=[0,2,1], conjugate=True)
#         Z = Uc - U + Vc @ V 
#         Zi = tf.linalg.inv(I+Z) #tf.linalg.solve(I+Z, I)
#         R = tf.concat([Zi @ (I-Z), -2 * Zi @ Vc], axis=2)

#     return R

def cayley_vec(W, x):
    _, m, n = W.shape
    if m >= n:
        I = tf.eye(n, dtype=W.dtype)[None,:,:]
        U, V = W[:, :n, :], W[:, n:, :]
        Uc = tf.transpose(U, perm=[0,2,1], conjugate=True)
        Vc = tf.transpose(V, perm=[0,2,1], conjugate=True)
        Z = U - Uc + Vc @ V 
        z = tf.linalg.solve(I+Z, x)
        y = tf.concat([(I-Z) @ z, -2 * V @ z], axis=1)
    else:
        I = tf.eye(m, dtype=W.dtype)[None,:,:]
        Uc, Vc = W[:, :, :m], W[:, :, m:]
        U = tf.transpose(Uc, perm=[0,2,1], conjugate=True)
        V = tf.transpose(Vc, perm=[0,2,1], conjugate=True)
        Z = Uc - U + Vc @ V 
        z = tf.concat([(I-Z), -2 * Vc], axis=2) @ x 
        y = tf.linalg.solve(I+Z, z)

    return y

class HouseholderLayer(keras.layers.Layer):
    def __init__(self, cin):
        super().__init__()
        self.cin = cin 
        self.v_dim = max(cin // 16, 10)
        Fq = keras.initializers.RandomNormal()((cin, self.v_dim))
        Fq = Fq / tf.norm(Fq, axis=0)
        self.Fq = tf.Variable(Fq, name="Fq", trainable=True)
        by = keras.initializers.Zeros()((cin,))
        self.by = tf.Variable(by, name="by", trainable=True)

    def call(self, x):
        for k in range(self.v_dim):
            v = tf.reshape(self.Fq[:, k], (self.cin, 1))
            v = v / tf.norm(v)
            x = x - 2 * (x @ v) @ tf.transpose(v)

        x += self.by 
        return x 
        
class ConvMonLipNet(keras.layers.Layer):
    def __init__(self, x_shape, channels, l2, kernel_size=3, mu=0.1, nu=10.):
        super().__init__()
        batch_size, image_size, _, cin = x_shape
        self.cin = cin
        self.channels = channels 
        self.kernel_size = kernel_size
        self.image_size = image_size 
        self.mu = mu 
        self.nu = nu 
        self.l2 = l2 
        self.batches = batch_size
        
        Fq = keras.initializers.GlorotUniform()((sum(channels), cin, kernel_size, kernel_size))
        self.Fq = tf.Variable(Fq, name="Fq", trainable=True)
        by = keras.initializers.Zeros()((cin,))
        self.by = tf.Variable(by, name="by", trainable=True)
        self.shift_matrix = None
        self.fft_shift_matrix()

        nz_1 = 0
        Fr, b = [], []
        for k, nz in enumerate(channels):
            R = keras.initializers.GlorotUniform()((nz, nz+nz_1, kernel_size, kernel_size))
            Fr.append(tf.Variable(R, name=f"Fr{k}", trainable=True))
            bk = keras.initializers.Zeros()((nz,))
            b.append(tf.Variable(bk, name=f"b{k}", trainable=True))
            nz_1 = nz 

        self.Fr = Fr
        self.b = b
        
    
    def fft_shift_matrix(self):
        if self.shift_matrix == None:
            n = self.image_size 
            s = -((self.kernel_size - 1) // 2)
            shift = np.reshape(np.arange(0, n), (1, n))
            shift = np.repeat(shift, repeats=n, axis=0)
            shift = shift + np.transpose(shift)
            shift = np.exp(1j * 2 * np.pi * s * shift / n)
            shift = tf.convert_to_tensor(shift, dtype=tf.complex64)
            shift = shift[:, :(n//2 + 1)]
            shift = tf.reshape(shift, (n * (n//2 + 1), 1, 1))
            self.shift_matrix = shift 

    def fft_weight(self, W):
        n = self.image_size 
        cout, cin, _, _ = W.shape 
        Wfft = tf.signal.rfft2d(W, fft_length=(n,n))
        Wfft = tf.reshape(Wfft, (cout, cin, n * (n // 2 + 1)))
        Wfft = tf.transpose(Wfft, perm=[2,0,1], conjugate=True)
        Wfft = self.shift_matrix * Wfft
        return Wfft

    def fft_vector(self, x):
        n = self.image_size 
        batches = self.batches
        cin = x.shape[1]
        # batches, cin, n, _ = x.shape
        xfft = tf.signal.rfft2d(x, fft_length=(n, n))
        xfft = tf.transpose(xfft, perm=[2, 3, 1, 0])
        xfft = tf.reshape(xfft, (n * (n // 2 + 1), cin, batches))
        return xfft 
    
    def ifft_vector(self, xfft):
        n = self.image_size 
        _, channels, batches = xfft.shape 
        xfft = tf.reshape(xfft, (n, n//2+1, channels, batches))
        xfft = tf.transpose(xfft, perm=[3, 2, 0, 1])
        x = tf.signal.irfft2d(xfft)
        return x 
    
    def l2loss(self):
        loss = tf.norm(self.Fq) ** 2
        for Fr in self.Fr:
            loss += tf.norm(Fr) ** 2
        return loss * self.l2 
    
    def call(self, x):
        sqrt_gam = (self.nu - self.mu) ** 0.5
        sqrt_2 = (2.) ** 0.5
        Fqfft = self.fft_weight(self.Fq)
        # Qfft = cayley(Fqfft)

        x = sqrt_gam * tf.transpose(x, perm=[0, 3, 1, 2])
        xhfft = self.fft_vector(x)
        xhfft = cayley_vec(Fqfft, xhfft)
        # xhfft = Qfft @ xhfft
        yhfft = []
        hk_1fft = xhfft[:, :0, :]
        idx = 0
        for k, nz in enumerate(self.channels):
            xkfft = xhfft[:, idx:idx+nz, :]
            Frfft = self.fft_weight(self.Fr[k])
            # Rfft = cayley(Frfft)
            ghfft = tf.concat([xkfft, hk_1fft], axis=1)
            ghfft = cayley_vec(Frfft, ghfft)
            # ghfft = Rfft @ ghfft 
            gh = self.ifft_vector(ghfft)
            gh = sqrt_2 * tf.nn.relu(sqrt_2 * gh + self.b[k][:, None, None])
            ghfft = self.fft_vector(gh)
            ghfft = cayley_vec(tf.transpose(Frfft, perm=[0,2,1], conjugate=True), ghfft)
            # ghfft = tf.transpose(Rfft, perm=[0,2,1], conjugate=True) @ ghfft
            hkfft = ghfft[:, :nz, :] - xkfft
            gkfft = ghfft[:, nz:, :]
            yhfft.append(hk_1fft-gkfft)
            idx += nz 
            hk_1fft = hkfft 
        yhfft.append(hk_1fft)
        yhfft = cayley_vec(tf.transpose(Fqfft, perm=[0,2,1], conjugate=True), tf.concat(yhfft, axis=1)) 
        # yhfft = tf.transpose(Qfft, perm=[0,2,1], conjugate=True) @ tf.concat(yhfft, axis=1)
        yh = self.ifft_vector(yhfft)
        y = 0.5 * ((self.mu + self.nu) * x + sqrt_gam * yh) + self.by[:, None, None] 
        y = tf.transpose(y, perm=[0, 2, 3, 1])
        return y

class ConvBiLipNet(keras.layers.Layer):
    def __init__(self, x_shape, channels, l2, kernel_size=3, mu=0.1, nu=10.):
        super().__init__()
        self.mon_layer1 = ConvMonLipNet(x_shape, channels, l2, kernel_size=kernel_size, mu=mu, nu=nu)
        self.ort_layer1 = HouseholderLayer(x_shape[-1])

    def l2loss(self):
        return self.mon_layer1.l2loss()
    
    def call(self, x):
        x = self.mon_layer1(x)
        x = self.ort_layer1(x)

        return x
    
# Wide-Resnet components
BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)

def make_random_feature_initializer(random_feature_type):
    # Use stddev=0.05 to replicate the default behavior of
    # tf.keras.initializer.RandomNormal.
    if random_feature_type == 'orf':
        return ed.OrthogonalRandomFeatures(stddev=0.05)
    elif random_feature_type == 'rff':
        return tf.keras.initializers.RandomNormal(stddev=0.05)
    else:
        return random_feature_type

def make_conv2d_layer(use_spec_norm,
                      spec_norm_iteration,
                      spec_norm_bound):
  """Defines type of Conv2D layer to use based on spectral normalization."""
  Conv2DBase = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=3,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal')

  def Conv2DNormed(*conv_args, **conv_kwargs):
    return ed.SpectralNormalizationConv2D(
        Conv2DBase(*conv_args, **conv_kwargs),
        iteration=spec_norm_iteration,
        norm_multiplier=spec_norm_bound)

  return Conv2DNormed if use_spec_norm else Conv2DBase

def apply_dropout(input_shape, dropout_rate, use_mc_dropout):
    """Applies a filter-wise dropout layer to the inputs."""
    logging.info('apply_dropout input shape %s', input_shape)
    if use_mc_dropout:
        dropout_layer = tf.keras.layers.Dropout(
            dropout_rate, noise_shape=[input_shape[0], 1, 1, input_shape[3]], training=True)
    else:
        dropout_layer = tf.keras.layers.Dropout(
            dropout_rate, noise_shape=[input_shape[0], 1, 1, input_shape[3]])

    return dropout_layer

class cftn_gp(keras.Model):
    def __init__(self, input_shape, batch_size, mu, nu, width_multiplier, num_classes,
                l2, use_mc_dropout, use_filterwise_dropout,dropout_rate, 
                gp_input_dim, gp_hidden_dim,
                gp_scale, gp_bias, gp_input_normalization,
                gp_random_feature_type, gp_cov_discount_factor,
                gp_cov_ridge_penalty,
                use_spec_norm, 
                spec_norm_iteration,
                spec_norm_bound_0,
                spec_norm_bound):
        super().__init__()
        Conv2D_0 = make_conv2d_layer(use_spec_norm,  
                            spec_norm_iteration,
                            spec_norm_bound_0)
        Conv2D = make_conv2d_layer(use_spec_norm,  
                            spec_norm_iteration,
                            1.0)
        
        GaussianProcess = functools.partial(  
                            ed.RandomFeatureGaussianProcess,
                            num_inducing=gp_hidden_dim,
                            gp_kernel_scale=gp_scale,
                            gp_output_bias=gp_bias,
                            normalize_input=gp_input_normalization,
                            gp_cov_momentum=gp_cov_discount_factor,
                            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
                            use_custom_random_features=True,
                            custom_random_features_initializer=make_random_feature_initializer(
                                gp_random_feature_type)
                            )
        image_size, _, cin = input_shape
        x_shape = (batch_size, image_size, image_size, cin)
        base_width = 16
        self.conv0 = Conv2D_0(base_width,strides=1,kernel_regularizer=tf.keras.regularizers.l2(l2))

        filters = base_width * width_multiplier 
        self.conv1 = Conv2D(filters, strides=1, kernel_regularizer=tf.keras.regularizers.l2(l2))
        x_shape = (batch_size, image_size, image_size, filters)
        # self.dp1 = apply_dropout(x_shape, dropout_rate, use_mc_dropout)
        self.cbln1 = ConvBiLipNet(x_shape=x_shape,
                                channels=[filters//8]*8,
                                l2=l2,
                                mu=mu,
                                nu=nu)
        
        filters = 2 * base_width * width_multiplier 
        self.conv2 = Conv2D(filters, strides=2, kernel_regularizer=tf.keras.regularizers.l2(l2))
        image_size = image_size // 2
        x_shape = (batch_size, image_size, image_size, filters)
        # self.dp2 = apply_dropout(x_shape, dropout_rate, use_mc_dropout)
        self.cbln2 = ConvBiLipNet(x_shape=x_shape,
                                   channels=[filters//8]*8,
                                   l2=l2,
                                   mu=mu,
                                   nu=nu)

        filters = 4 * base_width * width_multiplier 
        self.conv3 = Conv2D(filters, strides=2, kernel_regularizer=tf.keras.regularizers.l2(l2))
        image_size = image_size // 2
        x_shape = (batch_size, image_size, image_size, filters)
        # self.dp3 = apply_dropout(x_shape, dropout_rate, use_mc_dropout)

        self.cbln3 = ConvBiLipNet(x_shape=x_shape,
                                   channels=[filters//8]*8,
                                   l2=l2,
                                   mu=mu,
                                   nu=nu)
        
        self.bn1 = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                        gamma_regularizer=tf.keras.regularizers.l2(l2))
        self.act1 = tf.keras.layers.Activation('relu')
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=8)
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(
                        gp_input_dim,
                        kernel_initializer='random_normal',
                        use_bias=False,
                        trainable=False)
        self.gp = GaussianProcess(num_classes)

    def call(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.cbln1(x)
        x = self.conv2(x)
        x = self.cbln2(x)
        x = self.conv3(x)
        x = self.cbln3(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.dense(x)
        x = self.gp(x)
        return x
    
    def l2_loss(self):
        loss  = sum(self.bn1.losses)
        loss += sum(self.conv0.losses)
        loss += sum(self.conv1.losses)
        loss += sum(self.conv2.losses)
        loss += sum(self.conv3.losses)
        # loss += self.cbln1.l2loss()
        # loss += self.cbln2.l2loss()
        # loss += self.cbln3.l2loss()
        return loss
    
if __name__ == "__main__":
    batches = 2
    cin = 3 
    image_size = 32
    num_classes = 10 
    gp_input_dim = 128
    gp_hidden_dim = 1024
    gp_scale = 2.
    gp_bias = 0. 
    gp_input_normalization = True
    gp_random_feature_type = 'orf' 
    gp_cov_discount_factor = 1.
    gp_cov_ridge_penalty = -1.
    use_spec_norm = True
    spec_norm_iteration = 1
    spec_norm_bound = 6.0
    use_mc_dropout = False
    use_filterwise_dropout = True
    dropout_rate = 0.1
    l2 = 3e-4

    x = tf.random.normal((batches, image_size, image_size, cin))
    model = cftn_gp(
        input_shape=(image_size, image_size, cin), 
        batch_size=batches,
        num_classes=num_classes, 
        mu=0.25, 
        nu=10.0, 
        width_multiplier=8,
        l2=l2, 
        use_mc_dropout=use_mc_dropout, 
        use_filterwise_dropout=use_filterwise_dropout,
        dropout_rate=dropout_rate,
        gp_input_dim=gp_input_dim, 
        gp_hidden_dim=gp_hidden_dim,
        gp_scale=gp_scale, 
        gp_bias=gp_bias, 
        gp_input_normalization=gp_input_normalization,
        gp_random_feature_type=gp_random_feature_type, 
        gp_cov_discount_factor=gp_cov_discount_factor,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        use_spec_norm=use_spec_norm, 
        spec_norm_iteration=spec_norm_iteration,
        spec_norm_bound_0=spec_norm_bound,
        spec_norm_bound=spec_norm_bound
    )
    # print('Model input shape: %s', net.input_shape)
    # print('Model output shape: %s', net.output_shape)
    model.build((batches, image_size, image_size, cin))
    print('Model number of weights: %s', model.count_params())
    y = model(x)
    l2_loss = model.l2_loss()
    model.save('./test')
    
    
    