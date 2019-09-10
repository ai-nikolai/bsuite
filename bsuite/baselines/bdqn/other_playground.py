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


####################################################
# CODE
####################################################

class BayesianMLP():
  """A simple multilayer perceptron which flattens all non-batch dimensions."""

  def __init__(self, output_sizes):
    super(BayesianMLP, self).__init__()
    self._output_sizes = output_sizes

    self._model = tf.keras.Sequential([
        BatchFlatten(),
        tf.keras.layers(tfp.layers.MultivariateNormalTriL.params_size(self._output_sizes), activation=tf.nn.relu),
        tfp.layers.MultivariateNormalTriL(self._output_sizes),
    ])

    return self._model

def load_dataset(n=150, n_tst=150):
  #@title Synthesize dataset.
  w0 = 0.125
  b0 = 5.
  x_range = [-20, 60]
  np.random.seed(43)
  def s(x):
    g = (x - x_range[0]) / (x_range[1] - x_range[0])
    return 3 * (0.25 + g**2.)
  x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
  eps = np.random.randn(n) * s(x)
  y = (w0 * x * (1. + np.sin(x)) + b0) + eps
  x = x[..., np.newaxis]
  x_tst = np.linspace(*x_range, num=n_tst).astype(np.float32)
  x_tst = x_tst[..., np.newaxis]
  return y, x, x_tst


def build_raw_model():
  # Build model.
  model = tf.keras.Sequential([
  tf.keras.layers.Dense(1),
      tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t, scale=1)),
  ])
  return model

def build_model():

  model = build_raw_model()

  # Do inference.
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05), loss=negloglik)
  # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05), loss=negloglik)

  return model





####################################################
# MAIN
####################################################

if __name__=="__main__":

    # Load data.
    y, x, x_tst = load_dataset()

    # Specify model.
    model = build_model()
    model2 = build_raw_model()

    # Fit model given data.
    model.fit(x, y, epochs=500, verbose=False)

    yhat = model(x_tst)
    print(tf.squeeze( yhat.sample(1) ) )

    yhat = model2(x_tst)
    print(tf.squeeze( yhat.sample(1) ) )    # Pretend to load synthetic data set.
    # features = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
    # labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()
    #
    # # Specify model.
    # model = tfp.glm.Bernoulli()
    #
    # # Fit model given data.
    # coeffs, linear_response, is_converged, num_iter = tfp.glm.fit(
    #     model_matrix=features[:, tf.newaxis],
    #     response=tf.cast(labels, dtype=tf.float32),
    #     model=model)
    # print(coeffs)
    # ==> coeffs is approximately [1.618] (We're golden!)
# EOF
