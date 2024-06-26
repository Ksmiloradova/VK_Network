"""Generates the k-fold validation splits."""

import os

from absl import app
from absl import flags

import data_utils


_DATA_ROOT = flags.DEFINE_string(
    'data_root', None, required=True,
    help='Path containing the downloaded data.')


_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, required=True,
    help='Output directory to write the splits to')


def main(argv):
  del argv
  array_dict = data_utils.get_arrays(
      data_root=_DATA_ROOT.value,
      return_pca_embeddings=False,
      return_adjacencies=False)

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
  data_utils.generate_k_fold_splits(
      train_idx=array_dict['train_indices'],
      valid_idx=array_dict['valid_indices'],
      output_path=_OUTPUT_DIR.value,
      num_splits=data_utils.NUM_K_FOLD_SPLITS)


if __name__ == '__main__':
  app.run(main)
