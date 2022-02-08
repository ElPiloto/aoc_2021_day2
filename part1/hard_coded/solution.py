from typing import List, Optional, Tuple
import numpy as np


INPUT_FILE = '../aoc_input.txt'
cmd_to_vector = {
    'forward': np.array([1., 0.]),
    'up': np.array([0., -1.]),
    'down': np.array([0., 1.]),
}

def parse_line(x: str) -> Tuple[str, int]:
  """Returns command and magnitude from line."""
  cmd, magnitude = x.split(' ')
  return cmd, int(magnitude)


def read_file(fname: Optional[str] = INPUT_FILE) -> List[str]:
  with open(fname) as file:
    lines = [line.rstrip() for line in file]
  return lines


def solve(lines: List[str]):
  pos = np.zeros(shape=(2), dtype=np.float32)
  for l in lines:
    cmd, magnitude = parse_line(l)
    delta = cmd_to_vector[cmd]
    pos += delta * magnitude
  product = np.prod(pos)
  print(f'Final position: {pos}, product: {product}')


if __name__ == '__main__':
  lines = read_file()
  solve(lines)
