from enum import Enum
from typing import Optional
import numpy as np
import tensorflow as tf


class Commands(Enum):
  FWD = 0
  UP = 1
  DOWN = 2

  @classmethod
  def values(cls):
    return list(map(lambda c: c.value, cls))


class SyntheticGenerator():
  """Generates synthetic values."""

  def __init__(self,
      min_pos: int = 50,
      max_pos: int = 400,
      min_magnitude: int = 0,
      max_magnitude: int = 20,
      rng_seed: Optional[int] = 112233):
    self.rng_state = np.random.RandomState(rng_seed)
    self.min_pos = min_pos
    self.max_pos = max_pos
    self.min_magnitude = min_magnitude
    self.max_magnitude = max_magnitude

  def generator(self):
    def _generator():
      while True:
        # pick initial position
        pos = self.rng_state.randint(
            low = self.min_pos,
            high = self.max_pos,
            size=(2)
        )

        # pick command
        unique_commands = len(Commands.values())
        #command = self.rng_state.choice(range)
        command = self.rng_state.randint(
            low = 0,
            high = unique_commands,
            size=(1)
        )
        # TODO(elpiloto): Convert this to onehot.

        # pick magnitude
        magnitude = self.rng_state.randint(
            low = self.min_magnitude,
            high = self.max_magnitude,
            size=(1)
        )
        values = np.concatenate([pos, command, magnitude])
        values = values.astype(np.float32)
        # TODO(elpiloto): Add output here!!!
        yield values
    return _generator


class BatchDataset:

  def __init__(self, generator):
    self._generator = generator

  def __call__(self, batch_size: int):
    ds = tf.data.Dataset.from_generator(
            self._generator,
            (tf.float32),
            output_shapes=((4,)),
    )
    ds = ds.batch(batch_size=batch_size)
    return ds

if True:
  def integration():
    synthetic_generator = SyntheticGenerator()
    ds = BatchDataset(synthetic_generator.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=10).as_numpy_iterator()
  i = 0
  for i in range(10):
    print(next(batch_iterator))
  __import__('pdb').set_trace()
