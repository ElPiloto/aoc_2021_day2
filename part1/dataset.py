from enum import IntEnum
from typing import List, Optional
import numpy as np
import tensorflow as tf
from hard_coded import solution as hc_solution


AOC_INPUT_FILE = './aoc_input.txt'


class Commands(IntEnum):
  FORWARD = 0
  UP = 1
  DOWN = 2

  @classmethod
  def values(cls):
    return list(map(lambda c: c.value, cls))

CMD_TO_VECTOR = {
    Commands.FORWARD: np.array([1., 0.]),
    Commands.UP: np.array([0., -1.]),
    Commands.DOWN: np.array([0., 1.]),
}


def get_new_position(pos: np.ndarray, cmd: Commands, magnitude: float):
  """Returns new position after executing cmd."""
  delta = CMD_TO_VECTOR[cmd]
  new_pos = pos + delta * magnitude
  return new_pos


def command_idx_to_onehot(idx: int) -> np.ndarray:
  all_onehots = np.eye(3)
  command_onehot = np.squeeze(all_onehots[idx])
  return command_onehot


def solve_cumulative(lines: List[str]):
  inputs = []
  positions = []

  old_pos = np.zeros(shape=(2), dtype=np.float32)
  for l in lines:
    cmd, magnitude = l.split(' ')
    cmd = cmd.upper()
    cmd = Commands[cmd]
    cmd_idx = cmd.value
    cmd_onehot = command_idx_to_onehot(cmd_idx)
    magnitude = float(magnitude)

    # TODO(elpiloto): Convert cmd to onehot
    this_input = np.concatenate([np.copy(old_pos), cmd_onehot, [magnitude]])
    inputs.append(this_input)
    delta = CMD_TO_VECTOR[cmd]
    new_pos = delta * magnitude + old_pos
    positions.append(np.copy(new_pos))
    old_pos = new_pos
  return inputs, positions


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
        command_idx = self.rng_state.randint(
            low = 0,
            high = unique_commands,
            size=(1)
        )
        command_onehot = command_idx_to_onehot(command_idx)

        # pick magnitude
        magnitude = self.rng_state.randint(
            low = self.min_magnitude,
            high = self.max_magnitude,
            size=(1)
        )
        # [x, y, FWD_binary, UP_binary, DOWN_binary, magnitude]
        values = np.concatenate([pos, command_onehot, magnitude])
        values = values.astype(np.float32)

        # Get new position
        new_pos = get_new_position(
            values[:2],
            Commands(command_idx),
            values[-1],
        )
        yield values, new_pos
    return _generator


class AOCInputGenerator():

  def __init__(self, input_file: str = AOC_INPUT_FILE):
    lines = hc_solution.read_file(input_file)
    self._inputs, self._targets = solve_cumulative(lines)
    self._num_examples = len(self._inputs)

  def generator(self):
    def _generator():
      for i in range(self._num_examples):
        yield self._inputs[i], self._targets[i]
    return _generator


class BatchDataset:

  def __init__(self, generator):
    self._generator = generator

  def __call__(self, batch_size: int):
    ds = tf.data.Dataset.from_generator(
            self._generator,
            (tf.float32, tf.float32),
            output_shapes=((6,), (2,)),
    )
    ds = ds.batch(batch_size=batch_size)
    return ds

