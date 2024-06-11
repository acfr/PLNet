# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wide ResNet 28-10 with SNGP on CIFAR-10.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

## Reproducibility Instruction for CIFAR-100:

When running this script on CIFAR-100, set base_learning_rate=0.08 and
gp_mean_field_factor=12.5 to reproduce the benchmark result.

## Combining with MC Dropout:

As a single-model method, SNGP can be combined with other classic
uncertainty techniques (e.g., Monte Carlo dropout, deep ensemble) to further
improve performance.

This script supports adding Monte Carlo dropout to
SNGP by setting `use_mc_dropout=True`, setting `num_dropout_samples=10`
(or any integer larger than 1). Additionally we recommend adjust
`gp_mean_field_factor` slightly, since averaging already calibrated
individual models (in this case single SNGPs) can sometimes lead to
under-confidence [3].

## Note:

Different from the paper, this implementation computes the posterior using the
Laplace approximation based on the Gaussian likelihood (i.e., squared loss)
rather than that based on cross-entropy loss. As a result, the logits for all
classes share the same covariance. In the experiments, this approach is shown to
perform better and computationally more scalable when the number of output
classes are large.

## References:

[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
[3]: Rahul Rahaman, Alexandre H. Thiery. Uncertainty Quantification and Deep
     Ensembles.  _arXiv preprint arXiv:2007.08792_, 2020.
     https://arxiv.org/abs/2007.08792
[4]: Hendrycks, Dan et al. AugMix: A Simple Data Processing Method to Improve
     Robustness and Uncertainty. In _International Conference on Learning
     Representations_, 2020.
     https://arxiv.org/abs/1912.02781
[5]: Zhang, Hongyi et al. mixup: Beyond Empirical Risk Minimization. In
     _International Conference on Learning Representations_, 2018.
     https://arxiv.org/abs/1710.09412
"""

import functools
import os
import sys
import time
from absl import app
from absl import flags
from absl import logging
import warnings
import numpy as np

import edward as ed
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import utils  # local file import
from tensorboard.plugins.hparams import api as hp

import datasets
from cftn import cftn_gp

import scipy.io 

# Data Augmentation flags.
flags.DEFINE_bool('augmix', True,
                  'Whether to perform AugMix [4] on the input data.')
flags.DEFINE_integer('aug_count', 1,
                     'Number of augmentation operations in AugMix to perform '
                     'on the input image. In the simgle model context, it'
                     'should be 1. In the ensembles context, it should be'
                     'ensemble_size if we perform random_augment only; It'
                     'should be (ensemble_size - 1) if we perform augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
                     'Augmix depth, -1 meaning sampled depth. This corresponds'
                     'to line 7 in the Algorithm box in [4].')
flags.DEFINE_integer('augmix_width', 3,
                     'Augmix width. This corresponds to the k in line 5 in the'
                     'Algorithm box in [4].')
flags.DEFINE_float('mixup_alpha', 0.1, 'Mixup hyperparameter, 0. to diable.')

# Dropout flags
flags.DEFINE_bool('use_mc_dropout', False,
                  'Whether to use Monte Carlo dropout for the hidden layers.')
flags.DEFINE_bool('use_filterwise_dropout', True,
                  'Whether to use filterwise dropout for the hidden layers.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_integer('num_dropout_samples', 1,
                     'Number of dropout samples to use for prediction.')
flags.DEFINE_integer('num_dropout_samples_training', 1,
                     'Number of dropout samples for training.')


# SNGP flags.
flags.DEFINE_bool('use_spec_norm', True,
                  'Whether to apply spectral normalization.')
flags.DEFINE_integer(
    'spec_norm_iteration', 1,
    'Number of power iterations to perform for estimating '
    'the spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound_0', 6.,
                   'Upper bound to spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound', 0.95,
                   'Upper bound to spectral norm of weight matrices.')
flags.DEFINE_integer('width_multiplier', 8,
                  'width multiplier.')

# Gaussian process flags.
flags.DEFINE_bool('use_gp_layer', True,
                  'Whether to use Gaussian process as the output layer.')
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 2.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_input_dim', 128,
    'The dimension to reduce the neural network input for the GP layer '
    '(via random Gaussian projection which preserves distance by the '
    ' Johnson-Lindenstrauss lemma). If -1, no dimension reduction.')
flags.DEFINE_integer(
    'gp_hidden_dim', 1024,
    'The hidden dimension of the GP layer, which corresponds to the number of '
    'random features used for the approximation.')
flags.DEFINE_bool(
    'gp_input_normalization', True,
    'Whether to normalize the input using LayerNorm for GP layer.'
    'This is similar to automatic relevance determination (ARD) in the classic '
    'GP learning.')
flags.DEFINE_string(
    'gp_random_feature_type', 'orf',
    'The type of random feature to use. One of "rff" (random fourier feature), '
    '"orf" (orthogonal random feature).')
flags.DEFINE_float('gp_cov_ridge_penalty', 1.,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', -1.,
    'The discount factor to compute the moving average of precision matrix'
    'across epochs. If -1 then compute the exact precision matrix within the '
    'latest epoch.')
flags.DEFINE_float(
    'gp_mean_field_factor', 5.,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')

flags.DEFINE_float('mu', 0.2,
                   'Bi-Lipschitz group lower bound.')

flags.DEFINE_float('nu', 2.,
                   'Bi-Lipschitz group upper bound.')

# Redefining default values
# flags.FLAGS.set_default('base_learning_rate', 0.05)
# flags.FLAGS.set_default('l2', 3e-4)
flags.FLAGS.set_default('train_epochs', 200)
flags.FLAGS.set_default('corruptions_interval', 1)
# flags.FLAGS.set_default('dataset', "cifar100")
# flags.FLAGS.set_default('output_dir', "./results/cifar100/w5-snb0.35")
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg

  logging.info('Output dir at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)
  # Split the seed into a 2-tuple, for passing into dataset builder.
  dataset_seed = (FLAGS.seed, FLAGS.seed + 1)

  data_dir = utils.get_data_dir_from_flags(FLAGS)
  if FLAGS.use_gpu:
    logging.info('Use GPU')

  batch_size = (FLAGS.per_core_batch_size * FLAGS.num_cores
                // FLAGS.num_dropout_samples_training)
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  num_classes = 10 if FLAGS.dataset == 'cifar10' else 100

  aug_params = {
      'augmix': FLAGS.augmix,
      'aug_count': FLAGS.aug_count,
      'augmix_depth': FLAGS.augmix_depth,
      'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
      'augmix_width': FLAGS.augmix_width,
      'ensemble_size': 1,
      'mixup_alpha': FLAGS.mixup_alpha,
  }
  validation_proportion = 1. - FLAGS.train_proportion
  use_validation_set = validation_proportion > 0.
  if FLAGS.dataset == 'cifar10':
    dataset_builder_class = datasets.Cifar10Dataset
  else:
    dataset_builder_class = datasets.Cifar100Dataset
  # train_dataset_builder = dataset_builder_class(
  #     data_dir=data_dir,
  #     download_data=FLAGS.download_data,
  #     split=tfds.Split.TRAIN,
  #     use_bfloat16=FLAGS.use_bfloat16,
  #     aug_params=aug_params,
  #     validation_percent=validation_proportion,
  #     seed=dataset_seed)
  # train_dataset = train_dataset_builder.load(batch_size=batch_size)
  # train_sample_size = train_dataset_builder.num_examples
  if validation_proportion > 0.:
    validation_dataset_builder = dataset_builder_class(
        data_dir=data_dir,
        download_data=FLAGS.download_data,
        split=tfds.Split.VALIDATION,
        use_bfloat16=FLAGS.use_bfloat16,
        validation_percent=validation_proportion)
    validation_dataset = validation_dataset_builder.load(batch_size=batch_size)
    val_sample_size = validation_dataset_builder.num_examples
    steps_per_val = steps_per_epoch = int(val_sample_size / batch_size)
  clean_test_dataset_builder = dataset_builder_class(
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16)
  clean_test_dataset = clean_test_dataset_builder.load(
      batch_size=test_batch_size)

  steps_per_eval = clean_test_dataset_builder.num_examples // batch_size
  test_datasets = {'clean': clean_test_dataset}
  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar10':
      load_c_dataset = utils.load_cifar10_c
    else:
      load_c_dataset = functools.partial(utils.load_cifar100_c,
                                         path=FLAGS.cifar100_c_path)
    corruption_types, max_intensity = utils.load_corrupted_test_info(
        FLAGS.dataset)
    # for corruption in corruption_types:
    #   for intensity in range(1, max_intensity + 1):
    #     dataset = load_c_dataset(
    #         corruption_name=corruption,
    #         corruption_intensity=intensity,
    #         batch_size=test_batch_size,
    #         use_bfloat16=FLAGS.use_bfloat16)
    #     test_datasets['{0}_{1}'.format(corruption, intensity)] = (
    #         dataset)
        
  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
  
  model = tf.keras.models.load_model(f"{FLAGS.output_dir}/model")
  logging.info('Model number of weights: %s', model.count_params())

  metrics = {
      'test_negative_log_likelihood': tf.keras.metrics.Mean(),
      'test_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
      'test_ece': rm.metrics.ExpectedCalibrationError(
          num_bins=FLAGS.num_bins),
      'test_stddev': tf.keras.metrics.Mean(),
  }
  if use_validation_set:
    metrics.update({
        'val/negative_log_likelihood': tf.keras.metrics.Mean(),
        'val/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'val/ece': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
        'val/stddev': tf.keras.metrics.Mean(),
    })
  if FLAGS.corruptions_interval > 0:
    corrupt_metrics = {}
    for intensity in range(1, max_intensity + 1):
      for corruption in corruption_types:
        dataset_name = '{0}_{1}'.format(corruption, intensity)
        corrupt_metrics['test_nll_{}'.format(dataset_name)] = (
            tf.keras.metrics.Mean())
        corrupt_metrics['test_accuracy_{}'.format(dataset_name)] = (
            tf.keras.metrics.SparseCategoricalAccuracy())
        corrupt_metrics['test_ece_{}'.format(dataset_name)] = (
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
        corrupt_metrics['test_stddev_{}'.format(dataset_name)] = (
            tf.keras.metrics.Mean())

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      if dataset_name == "clean":
        images = inputs['features']
        labels = inputs['labels']
      else:
        images = inputs[0]
        labels = inputs[1]

      logits_list = []
      stddev_list = []
      for _ in range(FLAGS.num_dropout_samples):
        logits = model(images, training=False)
        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract both
          logits, covmat = logits
        else:
          covmat = tf.eye(FLAGS.per_core_batch_size)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        logits = ed.mean_field_logits(
            logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
        stddev = tf.sqrt(tf.linalg.diag_part(covmat))

        stddev_list.append(stddev)
        logits_list.append(logits)

      # Logits dimension is (num_samples, batch_size, num_classes).
      logits_list = tf.stack(logits_list, axis=0)
      stddev_list = tf.stack(stddev_list, axis=0)

      stddev = tf.reduce_mean(stddev_list, axis=0)
      probs_list = tf.nn.softmax(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)

      labels_broadcasted = tf.broadcast_to(
          labels, [FLAGS.num_dropout_samples, labels.shape[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits_list, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
          tf.math.log(float(FLAGS.num_dropout_samples)))

      if dataset_name == 'clean':
        metrics['test_negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test_accuracy'].update_state(labels, probs)
        metrics['test_ece'].add_batch(probs, label=labels)
        metrics['test_stddev'].update_state(stddev)
      elif dataset_name == 'val':
        metrics['val/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['val/accuracy'].update_state(labels, probs)
        metrics['val/ece'].add_batch(probs, label=labels)
        metrics['val/stddev'].update_state(stddev)
      else:
        corrupt_metrics['test_nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test_accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test_ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)
        corrupt_metrics['test_stddev_{}'.format(dataset_name)].update_state(
            stddev)

    for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
      step_fn(next(iterator))
      # strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test_ms_per_example': tf.keras.metrics.Mean()})

  # datasets_to_evaluate = {'clean': test_datasets['clean']}
  for dataset_name, test_dataset in test_datasets.items():
    test_iterator = iter(test_dataset)
    steps_per_eval = steps_per_val if dataset_name == 'val' else steps_per_eval
    test_start_time = time.time()
    test_step(test_iterator, dataset_name)
    ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
    metrics['test_ms_per_example'].update_state(ms_per_example)
    if dataset_name == "clean":
      acc = (metrics['test_accuracy'].result() * 100).numpy()
      ece = (metrics['test_ece'].result())['ece']
      nll = (metrics['test_negative_log_likelihood'].result()).numpy()
    else:
      acc = (corrupt_metrics[f'test_accuracy_{dataset_name}'].result() * 100).numpy()
      ece = (corrupt_metrics[f'test_ece_{dataset_name}'].result())['ece']
      nll = (corrupt_metrics[f'test_nll_{dataset_name}'].result()).numpy()

    msg = f"Acc: {acc:.1f}, ECE: {ece:.4f}, NLL: {nll:.4f} | {dataset_name}"
    logging.info(msg)

  for corruption in corruption_types:
      for intensity in range(1, max_intensity + 1):
        dataset_name = '{0}_{1}'.format(corruption, intensity)
        dataset = load_c_dataset(
            corruption_name=corruption,
            corruption_intensity=intensity,
            batch_size=test_batch_size,
            use_bfloat16=FLAGS.use_bfloat16)
        test_iterator = iter(dataset)
        steps_per_eval = steps_per_val if dataset_name == 'val' else steps_per_eval
        test_start_time = time.time()
        test_step(test_iterator, dataset_name)
        ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
        metrics['test_ms_per_example'].update_state(ms_per_example)
        if dataset_name == "clean":
          acc = (metrics['test_accuracy'].result() * 100).numpy()
          ece = (metrics['test_ece'].result())['ece']
          nll = (metrics['test_negative_log_likelihood'].result()).numpy()
        else:
          acc = (corrupt_metrics[f'test_accuracy_{dataset_name}'].result() * 100).numpy()
          ece = (corrupt_metrics[f'test_ece_{dataset_name}'].result())['ece']
          nll = (corrupt_metrics[f'test_nll_{dataset_name}'].result()).numpy()

        msg = f"Acc: {acc:.1f}, ECE: {ece:.4f}, NLL: {nll:.4f} | {dataset_name}"
        logging.info(msg)


  corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                    corruption_types,
                                                    max_intensity,
                                                    output_dir=FLAGS.output_dir)
  total_results = {name: metric.result() for name, metric in metrics.items()}
  total_results.update(corrupt_results)
  # Metrics from Robustness Metrics (like ECE) will return a dict with a
  # single key/value, instead of a scalar.
  total_results = {
      k: (list(v.values())[0] if isinstance(v, dict) else v)
      for k, v in total_results.items()
  }
  with open(f'{FLAGS.output_dir}/eval_result.txt', "w") as f:
    sys.stdout = f
    for k, v in total_results.items():
      msg = f'{k}: {v:.4f}'
      print(msg)
  scipy.io.savemat(f'{FLAGS.output_dir}/eval_result.mat',total_results)

if __name__ == '__main__':
  app.run(main)