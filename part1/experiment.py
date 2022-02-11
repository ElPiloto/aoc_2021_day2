"""Jaxline experiment for solving Day 2, Part Advent of Code."""
import functools
from typing import Dict, Optional

from absl import app
from absl import logging
from absl import flags
import haiku as hk
from haiku import data_structures
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
import ml_collections
import numpy as np
import optax
import tensorflow as tf
import wandb

import dataset

np.set_printoptions(suppress=True, precision=5)
tf.config.set_visible_devices([], 'GPU')
FLAGS = flags.FLAGS

# This may be unnecessary, try removing it when you have free time.
jax.config.update('jax_platform_name', 'gpu')

run = wandb.init(project='aoc_2021_day2_part1', entity='elpiloto')


def get_config():
  # Common config to all jaxline experiments.
  config = base_config.get_base_config()
  config.training_steps = 100000
  config.checkpoint_dir = './checkpoints/'
  # Needed because jaxline version from pypi is broken and version from github
  # breaks everything else.
  config.train_checkpoint_all_hosts = False
  config.interval_type = 'steps'

  # Our config options
  exp = config.experiment_kwargs = ml_collections.ConfigDict()
  exp.train_seed = 107993
  exp.eval_seed = 8802
  exp.learning_rate = 1e-8
  exp.batch_size = 256

  exp.data_config = ml_collections.ConfigDict()
  train = exp.data_config.train = ml_collections.ConfigDict()
  train.min_pos = 0
  train.max_pos = 2000
  train.min_magnitude = 0
  train.max_magnitude = 20

  eval = exp.data_config.eval = ml_collections.ConfigDict()
  eval.name = ["eval"]
  eval.min_pos = [0,]
  eval.max_pos = [2000,]
  eval.min_magnitude = [0,]
  eval.max_magnitude = [20,]

  model = exp.model = ml_collections.ConfigDict()
  model.output_sizes = [256, 256, 2]
  model.activation_fn = 'relu'
  wandb.config.update(exp.to_dict())
  return config


def mean_squared_error(params, model, inputs, targets):
  """Computes the mean squared error."""
  model_output = model.apply(params, inputs)
  # dimensions: [batch_size, 2]
  error = jnp.square(model_output - targets)
  summed = jnp.sum(error, axis=-1)
  # summed has shape: [batch_size]
  mse = jnp.mean(summed)
  return mse


def rounded_mean_squared_error(params, model, inputs, targets):
  """Computes the mean squared error."""
  model_output = model.apply(params, inputs)
  model_output = jnp.round(model_output)
  # dimensions: [batch_size, 2]
  error = jnp.square(model_output - targets)
  summed = jnp.sum(error, axis=-1)
  mse = jnp.mean(summed)
  return mse, model_output

class Experiment(experiment.AbstractExperiment):

  NON_BROADCAST_CHECKPOINT_ATTRS = {
       '_params': '_params',
       '_opt_state': '_opt_state',
  }

  def __init__(self,
                mode: str,
                train_seed: int,
                eval_seed: int,
                learning_rate: float,
                batch_size: int,
                data_config: ml_collections.ConfigDict,
                model: ml_collections.ConfigDict,
                init_rng: Optional[jnp.DeviceArray] = None):
      super().__init__(mode, init_rng=init_rng)
      self._mode = mode
      self._train_seed = train_seed
      self._eval_seed = eval_seed
      self._learning_rate = learning_rate
      self._data_config = data_config
      self._batch_size = batch_size
      self._config = get_config()
      self._model_config = model
      logging.log(logging.INFO, f'Launched experiment with mode = {mode}')
      run.tags += tuple(FLAGS.wandb_tags)
      self._counter = jnp.array([0.])

      # train and eval together
      if mode == 'train':
        # instantiate our training data
        self._train_data = self._build_train_data()
        # instantiate our evaluation data
        self._eval_datasets = self._build_eval_data()
        self._aoc_data = self._build_aoc_data()
        # instantiate our neural network
        model = self._initialize_model()
        train_inputs, _ = next(self._train_data)
        self._model = hk.without_apply_rng(hk.transform(model))
        self._params = self._model.init(
            init_rng,
            inputs=jnp.zeros_like(train_inputs)
        )
        # build our optimizer
        sched = optax.piecewise_constant_schedule(
            self._learning_rate,
            {
              10: 1.,
            }
        )
        # We put this in a optax schedule just for easy logging.
        self._sched = sched
        opt = optax.adam(learning_rate=sched)
        self._opt_state = opt.init(self._params)
        # Example output, I just like to keep this.
        _ = self._model.apply(self._params, train_inputs)
        # build our update fn, which is called by our step function

        # Make update function.
        def update_fn(params, inputs, targets):
          loss, grads = jax.value_and_grad(mean_squared_error)(params, self._model, inputs,
              targets)
          updates, opt_state = opt.update(grads, self._opt_state, params)
          params = optax.apply_updates(params, updates)
          return params, opt_state, loss
        self._update_fn = jax.jit(update_fn)

  def _initialize_model(self):
    activation = self._model_config.activation_fn
    try:
      activation_fn = getattr(jax.nn, activation)
    except:
      raise ValueError(f'Unknown activation function: {activation}')
    def _forward(inputs):
      output = hk.nets.MLP(
          output_sizes=self._model_config.output_sizes,
          activation=activation_fn,
          activate_final=True,
          )(inputs)
      return output
    return _forward



  def _build_train_data(self):
    ds_config = self._data_config['train']
    min_pos = ds_config['min_pos']
    max_pos = ds_config['max_pos']
    min_magnitude = ds_config['min_magnitude']
    max_magnitude = ds_config['max_magnitude']
    generator = dataset.SyntheticGenerator(
        min_pos=min_pos,
        max_pos=max_pos,
        min_magnitude=min_magnitude,
        max_magnitude=max_magnitude,
        rng_seed=self._train_seed,
    )
    ds = dataset.BatchDataset(generator.generator())
    batch_iterator = ds(batch_size=self._batch_size).as_numpy_iterator()
    return batch_iterator

  def _build_eval_data(self):
    """Builds eval data drawn from same distribution as training data."""
    ds_config = self._data_config['eval']
    datasets = {}
    eval_cfg_len = len(ds_config['name'])
    for i in range(eval_cfg_len):
      name = ds_config['name'][i]
      min_pos = ds_config['min_pos'][i]
      max_pos = ds_config['max_pos'][i]
      min_magnitude = ds_config['min_magnitude'][i]
      max_magnitude = ds_config['max_magnitude'][i]
      generator = dataset.SyntheticGenerator(
          min_pos=min_pos,
          max_pos=max_pos,
          min_magnitude=min_magnitude,
          max_magnitude=max_magnitude,
      )
      ds = dataset.BatchDataset(generator.generator())
      batch_iterator = ds(batch_size=self._batch_size).as_numpy_iterator()
      datasets[name] = batch_iterator
    return datasets

  def _build_aoc_data(self):
    # TODO(elpiloto): Figure out cleaner way of repeating dataset instead of
    # instantiating a new dataset each time
    generator = dataset.AOCInputGenerator()
    ds = dataset.BatchDataset(generator.generator())
    batch_iterator = ds(batch_size=100).as_numpy_iterator()
    return batch_iterator

  def step(self, *, global_step: jnp.ndarray, rng: jnp.ndarray, writer:
      Optional[jl_utils.Writer]) -> Dict[str, np.ndarray]:

    is_logging_step = global_step % 300 == 0

    # Get next training example
    inputs, targets = next(self._train_data)

    params, opt_state, loss = self._update_fn(self._params, inputs, targets)

    self._params = params
    self._opt_state = opt_state
    learning_rate = self._sched(global_step)[0]
    scalars = {
        'loss': loss,
        'learning_rate': learning_rate,
    }
    if is_logging_step and global_step > 299:
      eval_scalars = self.evaluate(global_step=global_step, rng=rng, writer=writer)
      scalars.update(eval_scalars)
      wandb.log(scalars, step=global_step)
      print(scalars)

    return scalars

  def evaluate(self, *, global_step: jnp.ndarray, rng: jnp.ndarray, writer:
      Optional[jl_utils.Writer]) -> Dict[str, np.ndarray]:

    errors = []
    for idx, (inputs, targets) in enumerate(self._aoc_data):
      error, predictions = rounded_mean_squared_error(self._params, self._model, inputs, targets)
      print(f'Error #{idx}: {error:.1f}')
      show_model_predictions(inputs, targets, predictions)
      errors.append(error)
    summed_error = np.sum(errors)
    self._aoc_data = self._build_aoc_data()
    return {'aoc_summed_error': summed_error}


def show_model_predictions(inputs, targets, predictions, num_examples=1):
  """Prints out model prediction versus ground truth."""
  for i in range(num_examples):
    prediction = predictions[i]
    target = targets[i]
    example = inputs[i]
    pos = example[:2]
    cmd_onehot = example[2:5]
    cmd_idx = np.argmax(cmd_onehot)
    cmd = dataset.Commands(cmd_idx).name
    magnitude = example[5]
    print(f'Input pos: {pos[0]}, {pos[1]}, {cmd} {magnitude} --> True: {target[0]}, {target[1]}, Model: {prediction[0]}, {prediction[1]}')
  
  


if __name__ == '__main__':
  flags.DEFINE_list('wandb_tags', [], 'Tags to send to wandb.')
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
