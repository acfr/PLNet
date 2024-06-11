import tensorflow as tf
from tensorflow.python import keras 

class DeepResNet(tf.keras.Model):
    """Defines a multi-layer residual network."""
    def __init__(self, num_classes, num_layers=3, num_hidden=128,
                dropout_rate=0.1, **classifier_kwargs):
        super().__init__()
        # Defines class meta data.
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.classifier_kwargs = classifier_kwargs

        # Defines the hidden layers.
        self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
        self.dense_layers = [self.make_dense_layer() for _ in range(num_layers)]

        # Defines the output layer.
        self.classifier = self.make_output_layer(num_classes)

    def call(self, inputs):
        # Projects the 2d input data to high dimension.
        hidden = self.input_layer(inputs)

        # Computes the ResNet hidden representations.
        for i in range(self.num_layers):
            resid = self.dense_layers[i](hidden)
            resid = tf.keras.layers.Dropout(self.dropout_rate)(resid)
            hidden += resid

        return self.classifier(hidden)

    def make_dense_layer(self):
        """Uses the Dense layer as the hidden layer."""
        return tf.keras.layers.Dense(self.num_hidden, activation="relu")

    def make_output_layer(self, num_classes):
        """Uses the Dense layer as the output layer."""
        return tf.keras.layers.Dense(
            num_classes, **self.classifier_kwargs)
    

def cayley(W):
    m, n = W.shape 
    if n > m:
        return tf.transpose(cayley(tf.transpose(W)))
    
    U, V = W[:n, :], W[n:, :]
    Z = U - tf.transpose(U) + tf.transpose(V) @ V 
    I = tf.eye(n)
    Zi = tf.linalg.solve(I+Z, I)
    R = tf.concat([Zi @ (I-Z), -2 * V @ Zi], 0)

    return R

# orthogonal layer based on HouseShoulder transformation
class UnitaryLayer(keras.layers.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim 
        b_shape = (input_dim,)
        b_value = keras.initializers.Zeros()(b_shape)
        self.b = tf.Variable(b_value, name="bias", trainable=True)
        # number of reflection vectors in HS transformation
        self.n_vec = max(input_dim // 8, 8)  
        v_value = keras.initializers.RandomNormal()((input_dim, self.n_vec))
        v_value = v_value / tf.norm(v_value, axis=0)
        self.v = tf.Variable(v_value, name="vf", trainable=True)

    def call(self, x, training=False):
        for k in range(self.n_vec):
            v = tf.reshape(self.v[:, k], (self.input_dim, 1))
            v = v / tf.norm(v)
            x = x - 2. * (x @ v) @ tf.transpose(v) 

        x += self.b    
        return x 
    
class MonLipNet(keras.layers.Layer):
    def __init__(self, input_dim, hidden_units, mu=0.1, nu=10.):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_units = hidden_units 
        L = len(hidden_units)
        self.mu = mu 
        self.nu = nu 
        Fq_shape = (input_dim, sum(hidden_units))
        Fq = keras.initializers.GlorotUniform()(Fq_shape)
        self.Fq = tf.Variable(Fq, name="Fq", trainable=True)
        fa = tf.reshape(tf.norm(Fq),(1,))
        self.fa = tf.Variable(fa, name="fa", trainable=True)

        by_shape = (input_dim,)
        by = keras.initializers.Zeros()(by_shape)
        self.by = tf.Variable(by, name="by", trainable=True)

        nz_1 = 0
        Fab, fab, b = [], [], []
        for k, nz in zip(range(L), hidden_units):
            b_v = keras.initializers.Zeros()((nz, ))
            b.append(tf.Variable(b_v, name=f'b{k}', trainable=True))
            Fab_v = keras.initializers.GlorotUniform()((nz+nz_1, nz))
            Fab.append(tf.Variable(Fab_v, name=f'Fab{k}', trainable=True))
            fab_v = tf.reshape(tf.norm(Fab_v),(1,))
            fab.append(tf.Variable(fab_v, name=f'fab{k}', trainable=True))
            nz_1 = nz

        self.Fab = Fab 
        self.fab = fab
        self.b = b

    def call(self, x, training=False):
        sqrt_2g, sqrt_g2 = tf.sqrt(2 * (self.nu - self.mu)), tf.sqrt((self.nu - self.mu)/2)
        y  = self.mu * x + self.by 
        Qt = cayley(self.fa / tf.norm(self.Fq) * self.Fq)
        idx = 0
        for k, nz in zip(range(len(self.hidden_units)), self.hidden_units):
            Qtk = Qt[:, idx:idx+nz]
            ABt = cayley((self.fab[k] / tf.norm(self.Fab[k])) * self.Fab[k])
            if k == 0:
                Atk = ABt
                St = Qtk @ Atk
                zk = tf.nn.relu(sqrt_2g * x @ St + self.b[k])
            else: 
                Atk, Btk = ABt[:nz, :], ABt[nz:, :]
                St = Qtk @ Atk - Qtk_1 @ Btk
                Vt = 2 * Ak_1 @ Btk 
                zk = tf.nn.relu(zk @ Vt + sqrt_2g * x @ St + self.b[k])
            y += sqrt_g2 * zk @ tf.transpose(St)
            Ak_1 = tf.transpose(Atk)
            Qtk_1 = Qtk
            idx =+ nz
            
        return y 
    
class BiLipNet(keras.layers.Layer):
    def __init__(self, input_dim, hidden_units, mu=0.1, nu=10.0):
        super().__init__()
        self.pre_rotate = UnitaryLayer(input_dim)
        self.mln = MonLipNet(input_dim, hidden_units, mu=mu, nu=nu)
        self.post_rotate = UnitaryLayer(input_dim)

    def call(self, x, training=False):
        x = self.pre_rotate(x, training=training)
        x = self.mln(x, training=training)
        x = self.post_rotate(x, training=training)
        return x 