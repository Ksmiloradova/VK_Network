"""Apply PCA to the users' BERT features.

Compute users' PCA features.
Recompute group and institution features from the user PCA features.
"""

import pathlib
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np

# pylint: disable=g-bad-import-order
import data_utils

Path = pathlib.Path

_NUMBER_OF_USERS_TO_ESTIMATE_PCA_ON = 1000000  # None indicates all.

FLAGS = flags.FLAGS
flags.DEFINE_string('data_root', None, 'Data root directory')


def _sample_vectors(vectors, num_samples, seed=0):
  """Randomly sample some vectors."""
  rand = np.random.RandomState(seed=seed)
  indices = rand.choice(vectors.shape[0], size=num_samples, replace=False)
  return vectors[indices]


def _pca(feat):
  """Returns evals (variances), evecs (rows are principal components)."""
  cov = np.cov(feat.T)
  _, evals, evecs = np.linalg.svd(cov, full_matrices=True)
  return evals, evecs

# Считывает numpy файл с эмбеддингами узлов 
# (пример: raw/node_feat.npy processed/user/node_feat.npy)
def _read_raw_user_features():
  """Load raw user features."""
  path = Path(FLAGS.data_root) / data_utils.RAW_NODE_FEATURES_FILENAME
  try:  # Use mmap if possible.
    features = np.load(path, mmap_mode='r')
  except FileNotFoundError:
    with open(path, 'rb') as fid:
      features = np.load(fid)
  return features


def _get_principal_components(features,
                              num_principal_components=129,
                              num_samples=10000,
                              seed=2,
                              dtype='f4'):
  """Estimate PCA features."""
  sample = _sample_vectors(
      features[:_NUMBER_OF_USERS_TO_ESTIMATE_PCA_ON], num_samples, seed=seed)
  # Compute PCA basis.
  _, evecs = _pca(sample)
  return evecs[:num_principal_components].T.astype(dtype)


def _project_features_onto_principal_components(features,
                                                principal_components,
                                                block_size=1000000):
  """Apply PCA iteratively."""
  num_principal_components = principal_components.shape[1]
  dtype = principal_components.dtype
  num_vectors = features.shape[0]
  num_features = features.shape[0]
  num_blocks = (num_features - 1) // block_size + 1
  pca_features = np.empty([num_vectors, num_principal_components], dtype=dtype)
  # Loop through in blocks.
  start_time = time.time()
  for i in range(num_blocks):
    i_start = i * block_size
    i_end = (i + 1) * block_size
    f = np.array(features[i_start:i_end].copy())
    pca_features[i_start:i_end] = np.dot(f, principal_components).astype(dtype)
    del f
    elapsed_time = time.time() - start_time
    time_left = elapsed_time / (i + 1) * (num_blocks - i - 1)
    logging.info('Features %d / %d. Elapsed time %.1f. Time left: %.1f', i_end,
                 num_vectors, elapsed_time, time_left)
  return pca_features


def _read_adjacency_indices():
  # Get adjacencies.
  return data_utils.get_arrays(
      data_root=FLAGS.data_root,
      use_fused_node_labels=False,
      use_fused_node_adjacencies=False,
      return_pca_embeddings=False,
  )


def _compute_group_pca_features(user_pca_features, index_arrays):
  return data_utils.user_features_to_group_features(
      index_arrays['group_user_index'], user_pca_features)


def _compute_institution_pca_features(group_pca_features, index_arrays):
  return data_utils.group_features_to_institution_features(
      index_arrays['institution_group_index'], group_pca_features)


def _write_array(path, array):
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, 'wb') as fid:
    np.save(fid, array)

# Обработка эмбеддингов узлов Методом Главных Компонент
# Всё вычисляется из первоначальных узлов
def main(unused_argv):
  data_root = Path(FLAGS.data_root)

  raw_user_features = _read_raw_user_features()   # читает "raw/node_feat.npy"
  principal_components = _get_principal_components(raw_user_features)
  # Схожий фрагмент
  user_pca_features = _project_features_onto_principal_components(
      raw_user_features, principal_components)
  del raw_user_features
  del principal_components

  user_pca_path = data_root / data_utils.PCA_USER_FEATURES_FILENAME
  group_pca_path = data_root / data_utils.PCA_GROUP_FEATURES_FILENAME
  institution_pca_path = (
      data_root / data_utils.PCA_INSTITUTION_FEATURES_FILENAME)
  merged_pca_path = data_root / data_utils.PCA_MERGED_FEATURES_FILENAME
  _write_array(user_pca_path, user_pca_features)

  # TODO: Сжать до получения инфы для пользователей сообществ
  # Схожий фрагмент
  # Compute group and institution features from user PCA features.
  index_arrays = _read_adjacency_indices()
  group_pca_features = _compute_group_pca_features(user_pca_features,
                                                     index_arrays)
  _write_array(group_pca_path, group_pca_features)

  # Схожий фрагмент
  merged_pca_features = np.concatenate(
      [user_pca_features, group_pca_features],
      axis=0)
  del group_pca_features
  _write_array(merged_pca_path, merged_pca_features)


if __name__ == '__main__':
  flags.mark_flag_as_required('data_root')
  app.run(main)
