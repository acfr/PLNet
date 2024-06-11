import os
import numpy as np 
import scipy.io
import tensorflow as tf
import official.nlp.modeling.layers as nlp_layers
from data import * 
from layer import DeepResNet, BiLipNet
import keras

class DeepResNetSNGP(DeepResNet):
    def __init__(self, **kwargs):
        self.spec_norm_bound = kwargs['spec_norm_bound']
        del kwargs['spec_norm_bound']
        super().__init__(**kwargs)

    def make_dense_layer(self):
        """Applies spectral normalization to the hidden layer."""
        dense_layer = super().make_dense_layer()
        return nlp_layers.SpectralNormalization(
            dense_layer, norm_multiplier=self.spec_norm_bound)

    def make_output_layer(self, num_classes):
        """Uses Gaussian process as the output layer."""
        return nlp_layers.RandomFeatureGaussianProcess(
            num_classes,
            gp_cov_momentum=-1,
            **self.classifier_kwargs)

    def call(self, inputs, training=False, return_covmat=False):
        # Gets logits and a covariance matrix from the GP layer.
        logits, covmat = super().call(inputs)

        # Returns only logits during training.
        if not training and return_covmat:
            return logits, covmat

        return logits

class ResetCovarianceCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        """Resets covariance matrix at the beginning of the epoch."""
        if epoch > 0:
            self.model.classifier.reset_covariance_matrix()

class DeepResNetSNGPWithCovReset(DeepResNetSNGP):
    def fit(self, *args, **kwargs):
        """Adds ResetCovarianceCallback to model callbacks."""
        kwargs["callbacks"] = list(kwargs.get("callbacks", []))
        kwargs["callbacks"].append(ResetCovarianceCallback())

        return super().fit(*args, **kwargs)

def compute_posterior_mean_probability(logits, covmat, lambda_param=np.pi / 8.):
    # Computes uncertainty-adjusted logits using the built-in method.
    logits_adjusted = nlp_layers.gaussian_process.mean_field_logits(
        logits, covmat, mean_field_factor=lambda_param)

    return tf.nn.softmax(logits_adjusted, axis=-1)[:, 0]

class BLNGP(keras.Model):
    """Defines a multi-layer residual network."""
    def __init__(self, num_classes, num_layers=6, num_hidden=128, num_features=64, mu=0.1, nu=10.0):
        super().__init__()
        # Defines class meta data.
        self.num_hidden = num_hidden
        self.num_layers = num_layers

        self.input_layer = tf.keras.layers.Dense(self.num_hidden, trainable=False)
        self.hidden_layer = BiLipNet(num_hidden, [num_features]*num_layers, mu=mu, nu=nu)
        self.classifier = nlp_layers.RandomFeatureGaussianProcess(num_classes, gp_cov_momentum=-1)

    def call(self, x, training=False, return_covmat=False):
        # Projects the 2d input data to high dimension.
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        logits, covmat = self.classifier(x)

        # Returns only logits during training.
        if not training and return_covmat:
            return logits, covmat

        return logits

class BLNGPCovReset(BLNGP):
    def fit(self, *args, **kwargs):
        """Adds ResetCovarianceCallback to model callbacks."""
        kwargs["callbacks"] = list(kwargs.get("callbacks", []))
        kwargs["callbacks"].append(ResetCovarianceCallback())

        return super().fit(*args, **kwargs)
     
Epochs = 600
Lr = 1e-4
root_dir = './results/NGP'
os.makedirs(root_dir, exist_ok=True)

for name in ['BiLipNet', 'SGNP']:
    # for snb in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for snb in [0.1, 0.3, 0.5, 0.7, 0.9]:
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
        train_examples, train_labels = make_training_data(sample_size=500)
        test_examples = make_testing_data()
        ood_examples = make_ood_data(sample_size=1000)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=Lr)
        train_config = dict(loss=loss, metrics=metrics, optimizer=optimizer)
        fit_config = dict(batch_size=128, epochs=Epochs)

        if name == 'SGNP':
            model = DeepResNetSNGPWithCovReset(num_classes=2, num_layers=3, num_hidden=128, spec_norm_bound=snb)
        
        if name == 'BiLipNet':
            mu, nu = (1 - snb) ** 3, (1 + snb) ** 3
            model = BLNGPCovReset(num_classes=2, num_layers=6, num_hidden=128, num_features=32, mu=mu, nu=nu)

        model.build((None, 2))
        model.summary()
        model.compile(**train_config)
        model.fit(train_examples, train_labels, verbose=0, **fit_config)

        sngp_logits, sngp_covmat = model(train_examples, return_covmat=True)
        train_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)
        train_uncert = (1 - train_probs) * train_probs / 0.25

        sngp_logits, sngp_covmat = model(ood_examples, return_covmat=True)
        ood_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)
        ood_uncert = (1 - ood_probs) * ood_probs / 0.25 

        sngp_logits, sngp_covmat = model(test_examples, return_covmat=True)
        test_probs = compute_posterior_mean_probability(sngp_logits, sngp_covmat)
        test_uncert = (1 - test_probs) * test_probs / 0.25 

        data = {
            'train_examples': train_examples,
            'train_labels': train_labels,
            'test_examples': test_examples,
            'ood_examples': ood_examples,
            'train_probs': train_probs.numpy(),
            'train_uncert': train_uncert.numpy(),
            'ood_probs': ood_probs.numpy(),
            'ood_uncert': ood_uncert.numpy(),
            'test_probs': test_probs.numpy(),
            'test_uncert': test_uncert.numpy()
        }

        train_dir = f'{root_dir}/{name}-snb{snb}'
        os.makedirs(train_dir, exist_ok=True)

        scipy.io.savemat(f'{train_dir}/data.mat', data)

        model.save_weights(f'{train_dir}/ckpt')