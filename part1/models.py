from typing import Callable
import inspect
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections


def remove_pos(inputs: jnp.ndarray) -> jnp.ndarray:
  """Removes positions from an input."""
  return inputs[:, 2:]

def get_activation(name: str) -> jax.custom_jvp:
  """Gets jax.nn activation fn from string."""
  try:
    activation_fn = getattr(jax.nn, name)
  except:
    raise ValueError(f'Unknown activation function: {name}')
  return activation_fn


def mlp(config: ml_collections.ConfigDict) -> Callable:
  """Returns an MLP."""
  def forward(inputs):
    if config.remove_pos:
      inputs = remove_pos(inputs)
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(inputs)
    return outputs
  return forward


def skip_connection_mlp(config: ml_collections.ConfigDict) -> Callable:
  """Returns an MLP with a skip connection from input pos to output."""
  def forward(inputs):
    orig_inputs = inputs
    if config.remove_pos:
      inputs = remove_pos(inputs)
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(inputs)
    return outputs + orig_inputs[:, :2]
  return forward


def skip_connection_mlp_scaled_magnitude(config: ml_collections.ConfigDict) -> Callable:
  """skip connection mlp that also scales command on input and output."""
  scale = config.magnitude_scale
  def forward(inputs):
    orig_inputs = inputs
    if config.remove_pos:
      inputs = remove_pos(inputs)
    # TODO(elpiloto): may be faster to just do a matmult with [0 0 0 0 0 1/scale.]
    new_inputs = jnp.concatenate(
        [inputs[:, :-1], inputs[:, -1:]/scale],
        axis=-1
    )
    outputs = hk.nets.MLP(
        output_sizes=config.output_sizes,
        activation=get_activation(config.activation_fn),
        activate_final=False)(new_inputs)
    return (scale * outputs) + orig_inputs[:, :2]
  return forward


def get_model(model_name: str, config: ml_collections.ConfigDict) -> hk.Module:
  """Uses model_name to build a specific model from this module."""
  local_fns = {k: v for k, v in globals().items() if inspect.isfunction(v)}
  if model_name in local_fns:
    return local_fns[model_name](config)
  raise ValueError(f'Could not find `{model_name}` function in model.py.')

