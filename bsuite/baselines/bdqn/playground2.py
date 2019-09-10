####################################################
#
# This Work is written by Nikolai Rozanov <nikolai>
#
# Contact:  nikolai.rozanov@gmail.com
#
# Copyright (C) 2018-Present Nikolai Rozanov
#
####################################################

####################################################
# IMPORT STATEMENTS
####################################################

# >>>>>>  Native Imports  <<<<<<<

# >>>>>>  Package Imports <<<<<<<
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# >>>>>>  Local Imports   <<<<<<<



# >>>>>>  Various Constants & Co.  <<<<<<<
# tf.enable_eager_execution()
tfd = tfp.distributions
negloglik = lambda y, p_y: -p_y.log_prob(y)
tf.keras.backend.set_floatx('float64')

####################################################
# CODE
####################################################
class BatchFlatten(tf.keras.Model):
  """A simple BatchFlatten"""
  def __init__(self):
    super(BatchFlatten, self).__init__()

  def call(self, inputs):
    dim = tf.reduce_prod(tf.shape(inputs)[1:])
    outputs = tf.reshape(inputs, [-1, dim])
    return outputs


class BayesianMLP(tf.keras.Model):
  """A simple multilayer perceptron which flattens all non-batch dimensions."""
  def __init__(self, output_sizes):
    super(BayesianMLP, self).__init__()
    self._output_sizes = output_sizes

    self.batch_flatten = BatchFlatten()
    self.hidden = tf.keras.layers.Dense(32, activation="relu")
    self.normal_in = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self._output_sizes))
    self.prob_out = tfp.layers.MultivariateNormalTriL(self._output_sizes)


  def call(self, inputs):
    outputs = self.batch_flatten(inputs)
    outputs = self.hidden(outputs)
    outputs = self.normal_in(outputs)
    outputs = self.prob_out(outputs)
    return outputs


class BayesianMLP2(tf.keras.Model):
  """A simple multilayer perceptron which flattens all non-batch dimensions."""
  def __init__(self, output_sizes):
    super(BayesianMLP2, self).__init__()
    self._output_sizes = output_sizes

    self._model = tf.keras.Sequential([
        BatchFlatten(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self._output_sizes)),
        tfp.layers.MultivariateNormalTriL(self._output_sizes)
    ])

  def call(self, inputs):
    outputs = self._model(inputs)
    return outputs

def load_dataset(n=100, dim_out=11):
  """creates a fake dataset"""
  x = np.linspace(-1,1,n)
  x = x[:,np.newaxis]
  x_out = x + np.random.randn(n, dim_out)
  y_out = x + np.random.rand(n,1)
  return x_out, y_out


def build_raw_model(dim_out=11):
  """Builds a raw model"""
  model = BayesianMLP2(dim_out)
  return model


def build_model():
  """ Builds the model with a loss function passed to it already. """
  model = build_raw_model()
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss=negloglik)
  return model



####################################################
# MAIN
####################################################

if __name__=="__main__":
    # Load data.
    out_dim=1

    #init of everything
    x, y = load_dataset(dim_out=out_dim)
    model2 = build_raw_model(out_dim)
    optimizer = tf.optimizers.Adam(learning_rate=1e-3)

    y = tf.convert_to_tensor(y)

    #prediction
    with tf.GradientTape() as tape:
        y_hat = model2(x)
        loss = -tf.reduce_sum( y_hat.log_prob(y) )
        print(loss)

    gradients = tape.gradient(loss, model2.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
    y_hat = model2(x)
    loss = -tf.reduce_sum( y_hat.log_prob(y) )
    print(loss)





# EOF
