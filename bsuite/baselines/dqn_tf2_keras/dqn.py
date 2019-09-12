# Copyright 2019 Tensorflow Authors & Nikolai Rozanov. All Rights Reserved.
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
"""A simple TensorFlow 2.0 -based DQN implementation, using Keras and gradient tape, as opposed to SNT.

Reference: "Playing atari with deep reinforcement learning" (Mnih et al, 2015).
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf.
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

class Dqn(base.Agent):
  """Deep Q-Networks."""

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
      seed: int = None,
  ):
    """A simple DQN agent."""
    # tf.keras.backend.set_floatx('float32')

    # DQN configuration and hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._optimizer = optimizer
    self._total_steps = 0
    self._total_episodes = 0
    self._replay = replay.Replay(capacity=replay_capacity)
    self._min_replay_size = min_replay_size
    tf.random.set_seed(seed)
    self._rng = np.random.RandomState(seed)
    self._epsilon_fn = lambda t: 10 / (10 + t)

    self._online_network = online_network
    self._target_network = target_network


  def policy(self, timestep: dm_env.TimeStep) -> int:
    """Select actions according to epsilon-greedy policy."""
    if self._rng.rand() < self._epsilon_fn(self._total_steps):
      action = self._rng.randint(self._num_actions)
    else:
      batched_obs = np.expand_dims(timestep.observation, axis=0)
      q_values = tf.squeeze(self._online_network(batched_obs))
      action = np.argmax(q_values)

    return np.int32(action)


  def update(self,
             old_step: dm_env.TimeStep,
             action: int,
             new_step: dm_env.TimeStep):
    """Takes in a transition from the environment."""
    self._total_steps += 1
    if new_step.last():
      self._total_episodes += 1

    if not old_step.last():
      self._replay.add(TransitionWithMaskAndNoise(
          x0=old_step.observation,
          a0=action,
          r1=new_step.reward,
          gamma1=new_step.discount,
          x1=new_step.observation,
      ))

    if self._replay.size < self._min_replay_size:
      return

    if self._total_steps % self._sgd_period == 0:
      x0,a0,r1,gamma1,x1  = self._replay.sample(self._batch_size)
      target = self._compute_target(r1,gamma1,x1)
      with tf.GradientTape() as tape:
        q0 = self._compute_prediction(x0,a0)
        td_error = target - q0
        loss = tf.reduce_sum( tf.square(td_error) )

      gradients = tape.gradient(loss, self._online_network.trainable_variables)
      self._optimizer.apply_gradients(zip(gradients, self._online_network.trainable_variables))

    if self._total_steps % self._target_update_period == 0:
      self._update_target_nets()

  def _compute_target(self, r1, gamma1, x1):
    """Computes the target Q of the target network"""
    q1 = tf.reduce_max( self._target_network(x1), axis=1 )
    r1 = tf.convert_to_tensor(r1)
    gamma1 = tf.convert_to_tensor(gamma1)
    target = r1 + gamma1 * self._discount * q1
    return target

  def _compute_prediction(self, x0, a0):
    """Computes the Q of the online network"""
    q0 = self._online_network(x0)
    qa0 = batched_index(q0, a0)
    return qa0

  def _update_target_nets(self):
    """Updates the target network from the online network"""
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
                  action_spec: dm_env.specs.DiscreteArray) -> Dqn:
  """Initialize a Bootstrapped DQN agent with default parameters."""
  online_network = MLP(output_sizes)
  target_network = MLP(output_sizes)
  return Dqn(
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
