# pylint: disable=g-bad-file-header
# Copyright 2019 Nikolai Rozanov. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple implementation of Bayesian Deep Q-Learning.

References:
1. "Efficient Exploration through Bayesian Deep Q-Networks" (Azizzadenesheli, et al., 2018)


Links:
1. [Paper]  https://arxiv.org/pdf/1802.04412.pdf
1. [Github] https://github.com/kazizzad/BDQN-MxNet-Gluon/blob/master/BDQN.ipynb

This implementation is potentially inefficient, in that it does not parallelize
computation, but it is much more readable and clear than complex TF ops.
"""

# Import all packages

import collections

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
import numpy as np
# import sonnet as
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Callable, Sequence

import logging
tf.get_logger().setLevel(logging.ERROR)

tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

class BayesianDqn(base.Agent):
  """Bayesian Deep Q-Networks with Gaussian Output."""

  def __init__(
      self,
      obs_spec: dm_env.specs.Array,
      action_spec: dm_env.specs.DiscreteArray,
      online_network: tf.keras.Sequential,
      target_network: tf.keras.Sequential,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: tf.keras.optimizers.Optimizer,
      posterior_optimizer: tf.optimizers.Optimizer = tf.optimizers.SGD(learning_rate=1e-2),
      seed: int = None,
  ):
    """A simple DQN agent."""
    # DQN configuration and hyperparameters.
    self._num_features = 32
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._optimizer = optimizer
    self._posterior_optimizer = posterior_optimizer
    self._total_steps = 0
    self._total_episodes = 0
    self._replay = replay.Replay(capacity=replay_capacity)
    self._min_replay_size = min_replay_size

    #time periods for updating
    self._sgd_period = sgd_period #the paper 4
    self._target_update_period = target_update_period #the paper 10000 (10k)
    self._posterior_update_period = 2000 #the paper 100000 (100k)
    self._sample_out_mus_period = 4 #the paper 1000 (1k)

    #neural network for the features
    self._online_network = online_network
    self._target_network = target_network

    # normal output distribution
    self._target_mus = []
    self._target_mu_covs = []
    self._normal_distros = []
    self._out_mus = []
    eye = tf.eye(self._num_features)
    bijector = tfp.bijectors.FillTriangular(upper=False)
    for idx in range(self._num_actions):
      mu = tf.random.normal([self._num_features]) #needs size: num_features
      cov = tf.random.normal([self._num_features, self._num_features], stddev=0.1) + eye #needs size: num_features x num_features
      cov = tf.linalg.band_part(cov,0,-1) #upper triangular
      cov = 0.5 * (cov + tf.transpose(cov) ) #make it symmetric
      chol = tf.linalg.cholesky(cov)
      chol = bijector.inverse(chol)
      mu_cov = tf.concat([mu,chol],0)
      mu_cov = tf.Variable(mu_cov)
      #
      normal_distro = tfp.layers.MultivariateNormalTriL(self._num_features)
      self._target_mus.append(normal_distro(mu_cov).mean())
      self._target_mu_covs.append(mu_cov)
      self._normal_distros.append(normal_distro)
      self._out_mus.append(normal_distro(mu_cov).sample())

    # setting unified keras backend
    tf.keras.backend.set_floatx('float32')

  def policy(self, timestep: dm_env.TimeStep) -> int:
    """Select actions according to epsilon-greedy policy."""
    batched_obs = np.expand_dims(timestep.observation, axis=0)
    q_values = self._compute_online_q(batched_obs)
    action = np.argmax(q_values)
    return np.int32(action)

  def update(self,
             old_step: dm_env.TimeStep,
             action: int,
             new_step: dm_env.TimeStep):
    """Takes in a transition from the environment."""
    #counting
    self._total_steps += 1
    if new_step.last():
      self._total_episodes += 1

    #adding data to the replay buffer
    if not old_step.last():
      self._replay.add(TransitionWithMaskAndNoise(
          x0=old_step.observation,
          a0=action,
          r1=new_step.reward,
          gamma1=new_step.discount,
          x1=new_step.observation,
      ))

    # Keep gathering data
    if self._replay.size < self._min_replay_size:
      return

    #training step
    if self._total_steps % self._sgd_period == 0:
      x0,a0,r1,gamma1,x1  = self._replay.sample(self._batch_size)
      target = self._compute_target(r1,gamma1,x1)
      with tf.GradientTape() as tape:
        q0 = self._compute_prediction(x0,a0)
        # loss = -tf.reduce_sum( q0.log_prob(target) ) #q0 is not a distribution
        td_error = target - q0
        loss = tf.reduce_sum( tf.square(td_error) )
      gradients = tape.gradient(loss, self._online_network.trainable_variables)
      self._optimizer.apply_gradients(zip(gradients, self._online_network.trainable_variables))

    #calculating posterior
    if self._total_steps % self._posterior_update_period == 0:
      self._compute_posterior()

    #sampling new out weights (mus)
    if self._total_steps % self._sample_out_mus_period == 0:
      self._sample_out_mus()

    #updating nets
    if self._total_steps % self._target_update_period == 0:
      self._update_target_nets()

  @tf.function
  def _compute_online_q(self, x0):
    """ Computes the value for each action in x0 batch. """
    features = self._online_network(x0)
    action_weights = tf.stack(self._out_mus) # num_actions x num_features
    q0 = features @ tf.transpose(action_weights)
    return q0

  @tf.function
  def _get_target_mus(self):
    """ Samples the out_mus. """
    for idx in range(self._num_actions):
      self._target_mus[idx] = self._normal_distros[idx](self._target_mu_covs[idx]).mean()

  @tf.function
  def _sample_out_mus(self):
    """ Samples the out_mus. """
    for idx in range(self._num_actions):
      self._out_mus[idx] = self._normal_distros[idx](self._target_mu_covs[idx]).sample()

  @tf.function
  def _compute_target(self, r1, gamma1, x1):
    """Computes the target Q of the target network. """
    features = self._target_network(x1)
    action_weights = tf.stack(self._target_mus) # num_actions x num_features
    q1 = tf.reduce_max( features @ tf.transpose(action_weights), axis=1 )
    r1 = tf.cast( tf.convert_to_tensor(r1), tf.float32 )
    gamma1 = tf.cast( tf.convert_to_tensor(gamma1), tf.float32 )
    target = r1 + gamma1 * self._discount * q1
    return target

  @tf.function
  def _compute_prediction(self, x0, a0):
    """Computes the Q of the online network. """
    features = self._online_network(x0) # batch x num_features
    action_weights = tf.stack(self._out_mus) # num_actions x num_features
    # a0 =  tf.convert_to_tensor(a0)
    batched_action_vectors = tf.gather(action_weights, a0) #batch x num_features
    qa0 = tf.reduce_sum( tf.multiply(batched_action_vectors, features), axis=1 )
    return qa0

  @tf.function
  def _compute_posterior(self):
    """ Computes the posterior on w and updates mu list. """
    x0,a0,r1,gamma1,x1 = self._replay.sample(self._batch_size)
    for idx in range(self._num_actions):
      mask = a0==idx
      mask = tf.convert_to_tensor(mask)
      mask = tf.cast(mask, tf.int32)
      x0a = tf.gather(x0,mask) #extracts only the x where a0==action
      phi = self._online_network(x0a)
      #approximate posterior update
      with tf.GradientTape() as tape:
        normal_distro = self._normal_distros[idx](self._target_mu_covs[idx])
        loss = -tf.reduce_sum( normal_distro.log_prob(phi) )
      gradients = tape.gradient(loss, normal_distro.variables)
      self._optimizer.apply_gradients(zip(gradients, normal_distro.variables))
    #need to update all the target mus
    self._get_target_mus()

  def _update_target_nets(self):
    """Updates the target network from the online network. """
    self._target_network.set_weights(self._online_network.get_weights())

TransitionWithMaskAndNoise = collections.namedtuple(
    'TransitionWithMaskAndNoise',
    ['x0', 'a0', 'r1', 'gamma1', 'x1'])


class BatchFlatten(tf.keras.Model):
  """A simple BatchFlatten"""
  def __init__(self):
    super(BatchFlatten, self).__init__()

  def call(self, inputs):
    dim = tf.reduce_prod(tf.shape(inputs)[1:])
    outputs = tf.reshape(inputs, [-1, dim])
    return outputs


class MLP(tf.keras.Model):
  """A simple multilayer perceptron which flattens all non-batch dimensions."""

  def __init__(self, output_sizes):
    super(MLP, self).__init__()
    self._output_sizes = output_sizes

    self._model = tf.keras.Sequential([
        BatchFlatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(self._output_sizes),
    ])

  def call(self, inputs):
    outputs = self._model(inputs)
    return outputs

# class BayesianMLP(tf.keras.Model):
#   """A simple multilayer perceptron which flattens all non-batch dimensions."""
#
#   def __init__(self, output_sizes):
#     super(BayesianMLP, self).__init__()
#     self._output_sizes = output_sizes
#
#     self._model = tf.keras.Sequential([
#         BatchFlatten(),
#         tf.keras.layers.Dense(32, activation="relu"),
#         tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self._output_sizes)),
#         tfp.layers.MultivariateNormalTriL(self._output_sizes)
#     ])
#
#   def call(self, inputs):
#     outputs = self._model(inputs)
#     return outputs


def batched_index(values, indices):
  """Equivalent to `values[:, indices]`.
  Performs indexing on batches and sequence-batches by reducing over
  zero-masked values. Compared to indexing with `tf.gather` this approach is
  more general and TPU-friendly, but may be less efficient if `num_values`
  is large. It works with tensors whose shapes are unspecified or
  partially-specified, but this op will only do shape checking on shape
  information available at graph construction time. When complete shape
  information is absent, certain shape incompatibilities may not be detected at
  runtime! See `indexing_ops_test` for detailed examples.
  Args:
    values: tensor of shape `[B, num_values]` or `[T, B, num_values]`
    indices: tensor of shape `[B]` or `[T, B]` containing indices.
  Returns:
    Tensor of shape `[B]` or `[T, B]` containing values for the given indices.
  Raises: ValueError if values and indices have sizes that are known
    statically (i.e. during graph construction), and those sizes are not
    compatible (see shape descriptions in Args list above).
  """
  values = tf.convert_to_tensor(values)
  indices = tf.convert_to_tensor(indices)

  one_hot_indices = tf.one_hot(
    indices,
    tf.shape(values)[-1],
    dtype=values.dtype
  )

  return tf.reduce_sum(values * one_hot_indices, axis=-1)




def default_agent(obs_spec: dm_env.specs.Array,
                  action_spec: dm_env.specs.DiscreteArray) -> BayesianDqn:
  """Initialize a Bootstrapped DQN agent with default parameters."""
  num_features = 32
  online_network = MLP(num_features)
  target_network = MLP(num_features)
  return BayesianDqn(
      obs_spec=obs_spec,
      action_spec=action_spec,
      online_network=online_network,
      target_network=target_network,
      batch_size=32,
      agent_discount=.99,
      replay_capacity=10000,
      min_replay_size=128,
      sgd_period=1,
      target_update_period=4,
      optimizer=tf.optimizers.Adam(learning_rate=1e-3),
      seed=42,
      )
