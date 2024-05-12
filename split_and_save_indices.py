
"""Split and save the train/valid/test indices.

Usage:

python3 split_and_save_indices.py --data_root="mag_data"
"""

import pathlib

from absl import app
from absl import flags
import numpy as np
import torch

Path = pathlib.Path


FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', None, 'Data root directory')
flags.DEFINE_string('root', 'ROOT', '')


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  mag_directory = Path(FLAGS.data_root)
  raw_directory = mag_directory / 'raw'
  raw_directory.parent.mkdir(parents=True, exist_ok=True)
  splits_dict = torch.load(str(mag_directory / 'split_dict.pt'))
  for key, indices in splits_dict.items():
    np.save(str(raw_directory / f'{key}_idx.npy'), indices)


if __name__ == '__main__':
  # print(flags.FLAGS)
  # flags.mark_flag_as_required('root')
  app.run(main)
