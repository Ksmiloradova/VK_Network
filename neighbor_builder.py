"""Find neighborhoods around user feature embeddings."""

import pathlib

from absl import app
from absl import flags
from absl import logging
import annoy
import numpy as np
import scipy.sparse as sp

# pylint: disable=g-bad-import-order
import data_utils

Path = pathlib.Path


_USER_USER_B_PATH = 'ogb_mag_adjacencies/user_user_b.npz'

FLAGS = flags.FLAGS
flags.DEFINE_string('data_root', None, 'Data root directory')

# Считывает "preprocessed/user_feat_pca_129.npy"
def _read_user_pca_features():
  data_root = Path(FLAGS.data_root)
  path = data_root / data_utils.PCA_USER_FEATURES_FILENAME
  with open(path, 'rb') as fid:
    return np.load(fid)


def _read_adjacency_indices():
  # Get adjacencies.
  return data_utils.get_arrays(
      data_root=FLAGS.data_root,
      use_fused_node_labels=False,
      use_fused_node_adjacencies=False,
      return_pca_embeddings=False,
  )


def build_annoy_index(features):
  """Build the Annoy index."""
  logging.info('Building annoy index')
  num_vectors, vector_size = features.shape
  annoy_index = annoy.AnnoyIndex(vector_size, 'euclidean')
  for i, x in enumerate(features):
    annoy_index.add_item(i, x)
    if i % 1000000 == 0:
      logging.info('Adding: %d / %d (%.3g %%)', i, num_vectors,
                   100 * i / num_vectors)
  n_trees = 10
  _ = annoy_index.build(n_trees)
  return annoy_index

# Начало куска annoy
def _get_annoy_index_path():
  return Path(FLAGS.data_root) / data_utils.PREPROCESSED_DIR / 'annoy_index.ann'


def save_annoy_index(annoy_index):
  logging.info('Saving annoy index')
  index_path = _get_annoy_index_path()
  index_path.parent.mkdir(parents=True, exist_ok=True)
  annoy_index.save(str(index_path))


def read_annoy_index(features):
  index_path = _get_annoy_index_path()
  vector_size = features.shape[1]
  annoy_index = annoy.AnnoyIndex(vector_size, 'euclidean')
  annoy_index.load(str(index_path))
  return annoy_index
# Конец куска annoy

def compute_neighbor_indices_and_distances(features):
  """Use the pre-built Annoy index to compute neighbor indices and distances."""
  logging.info('Computing neighbors and distances')
  annoy_index = read_annoy_index(features)
  num_vectors = features.shape[0]

  k = 20
  pad_k = 5
  search_k = -1
  neighbor_indices = np.zeros([num_vectors, k + pad_k + 1], dtype=np.int32)
  neighbor_distances = np.zeros([num_vectors, k + pad_k + 1], dtype=np.float32)
  for i in range(num_vectors):
    neighbor_indices[i], neighbor_distances[i] = annoy_index.get_nns_by_item(
        i, k + pad_k + 1, search_k=search_k, include_distances=True)
    if i % 10000 == 0:
      logging.info('Finding neighbors %d / %d', i, num_vectors)
  return neighbor_indices, neighbor_distances

# Сохранение готовых соседей
def _write_neighbors(neighbor_indices, neighbor_distances):
  """Write neighbor indices and distances."""
  logging.info('Writing neighbors')
  indices_path = Path(FLAGS.data_root) / data_utils.NEIGHBOR_INDICES_FILENAME
  distances_path = (
      Path(FLAGS.data_root) / data_utils.NEIGHBOR_DISTANCES_FILENAME)
  indices_path.parent.mkdir(parents=True, exist_ok=True)
  distances_path.parent.mkdir(parents=True, exist_ok=True)
  with open(indices_path, 'wb') as fid:
    np.save(fid, neighbor_indices)
  with open(distances_path, 'wb') as fid:
    np.save(fid, neighbor_distances)


def _write_fused_edges(fused_user_adjacency_matrix):
  """Write fused edges."""
  data_root = Path(FLAGS.data_root)
  edges_path = data_root / data_utils.FUSED_USER_EDGES_FILENAME
  edges_t_path = data_root / data_utils.FUSED_USER_EDGES_T_FILENAME
  edges_path.parent.mkdir(parents=True, exist_ok=True)
  edges_t_path.parent.mkdir(parents=True, exist_ok=True)
  with open(edges_path, 'wb') as fid:
    sp.save_npz(fid, fused_user_adjacency_matrix)
  with open(edges_t_path, 'wb') as fid:
    sp.save_npz(fid, fused_user_adjacency_matrix.T)


def _write_fused_nodes(fused_node_labels):
  """Write fused nodes."""
  labels_path = Path(FLAGS.data_root) / data_utils.FUSED_NODE_LABELS_FILENAME
  labels_path.parent.mkdir(parents=True, exist_ok=True)
  with open(labels_path, 'wb') as fid:
    np.save(fid, fused_node_labels)

# 
def main(unused_argv):
  user_pca_features = _read_user_pca_features()
  # Find neighbors.
  annoy_index = build_annoy_index(user_pca_features)
  save_annoy_index(annoy_index)
  # Распределение статей по соседству
  neighbor_indices, neighbor_distances = compute_neighbor_indices_and_distances(
      user_pca_features)
  del user_pca_features
  _write_neighbors(neighbor_indices, neighbor_distances)

  data = _read_adjacency_indices()
  user_user_csr = data['user_user_index']
  user_label = data['user_label']
  train_indices = data['train_indices']
  valid_indices = data['valid_indices']
  test_indices = data['test_indices']
  del data

  fused_user_adjacency_matrix = data_utils.generate_fused_user_adjacency_matrix(
      neighbor_indices, neighbor_distances, user_user_csr)
  _write_fused_edges(fused_user_adjacency_matrix)
  del fused_user_adjacency_matrix
  del user_user_csr

  fused_node_labels = data_utils.generate_fused_node_labels(
      neighbor_indices, neighbor_distances, user_label, train_indices,
      valid_indices, test_indices)
  _write_fused_nodes(fused_node_labels)


if __name__ == '__main__':
  flags.mark_flag_as_required('data_root')
  app.run(main)
