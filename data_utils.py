"""Dataset utilities."""

import functools
import pathlib
from typing import Dict, Tuple

from absl import logging
from graph_nets import graphs as tf_graphs
from graph_nets import utils_tf
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tqdm

# pylint: disable=g-bad-import-order
import sub_sampler

Path = pathlib.Path

NUM_USERS = 537_442_563
NUM_GROUPS = 18_630_018
# NUM_INSTITUTIONS = 
EMBEDDING_SIZE = 768 #(!)
NUM_CLASSES = 153

NUM_NODES = NUM_USERS + NUM_GROUPS
# NUM_EDGES = 1_728_364_232
assert NUM_NODES == 556_072_581

# (?)
NUM_K_FOLD_SPLITS = 10

# (!) twist maybe
OFFSETS = {
    "user": 0,
    "group": NUM_USERS,
    # "institution": NUM_GROUPS,
}


SIZES = {
    "user": NUM_USERS,
    "group": NUM_GROUPS,
    # "institution": NUM_INSTITUTIONS
}

RAW_DIR = Path("raw")
PREPROCESSED_DIR = Path("preprocessed")

RAW_NODE_FEATURES_FILENAME = RAW_DIR / "node_feat.npy"
RAW_NODE_LABELS_FILENAME = RAW_DIR / "node_label.npy"
RAW_NODE_YEAR_FILENAME = RAW_DIR / "node_year.npy"

# (!)
TRAIN_INDEX_FILENAME = RAW_DIR / "train_idx.npy"
VALID_INDEX_FILENAME = RAW_DIR / "train_idx.npy"
TEST_INDEX_FILENAME = RAW_DIR / "train_idx.npy"

EDGES_USER_USER_B = PREPROCESSED_DIR / "user_user_b.npz"
EDGES_USER_USER_B_T = PREPROCESSED_DIR / "user_user_b_t.npz"
EDGES_GROUP_INSTITUTION = PREPROCESSED_DIR / "group_user.npz"
EDGES_INSTITUTION_GROUP = PREPROCESSED_DIR / "user_group.npz"
EDGES_GROUP_USER = PREPROCESSED_DIR / "group_user.npz"
EDGES_USER_GROUP = PREPROCESSED_DIR / "user_group.npz"

PCA_USER_FEATURES_FILENAME = PREPROCESSED_DIR / "user_feat_pca_129.npy"
PCA_GROUP_FEATURES_FILENAME = (
    PREPROCESSED_DIR / "group_feat_from_user_feat_pca_129.npy")
PCA_INSTITUTION_FEATURES_FILENAME = (
    PREPROCESSED_DIR / "institution_feat_from_user_feat_pca_129.npy")
PCA_MERGED_FEATURES_FILENAME = (
    PREPROCESSED_DIR / "merged_feat_from_user_feat_pca_129.npy")

NEIGHBOR_INDICES_FILENAME = PREPROCESSED_DIR / "neighbor_indices.npy"
NEIGHBOR_DISTANCES_FILENAME = PREPROCESSED_DIR / "neighbor_distances.npy"

FUSED_NODE_LABELS_FILENAME = PREPROCESSED_DIR / "fused_node_labels.npy"
FUSED_USER_EDGES_FILENAME = PREPROCESSED_DIR / "fused_user_edges.npz"
FUSED_USER_EDGES_T_FILENAME = PREPROCESSED_DIR / "fused_user_edges_t.npz"

K_FOLD_SPLITS_DIR = Path("k_fold_splits")


def get_raw_directory(data_root):
  return Path(data_root) / "raw"


def get_preprocessed_directory(data_root):
  return Path(data_root) / "preprocessed"


def _log_path_decorator(fn):
  def _decorated_fn(path, **kwargs):
    logging.info("Loading %s", path)
    output = fn(path, **kwargs)
    logging.info("Finish loading %s", path)
    return output
  return _decorated_fn


@_log_path_decorator
def load_csr(path, debug=False):
  if debug:
    # Dummy matrix for debugging.
    return sp.csr_matrix(np.zeros([10, 10]))
  ret = sp.load_npz(str(path))
  print('load_csr ret shape', ret.shape)
  return ret


@_log_path_decorator
def load_npy(path):
  return np.load(str(path))


@functools.lru_cache()
def get_arrays(data_root="/data/",
               use_fused_node_labels=True,
               use_fused_node_adjacencies=True,
               return_pca_embeddings=True,
               k_fold_split_id=None,
               return_adjacencies=True,
               use_dummy_adjacencies=False):
  """Returns all arrays needed for training."""
  logging.info("Starting to get files")

  data_root = Path(data_root)

  array_dict = {}
  array_dict["user_year"] = load_npy(data_root / RAW_NODE_YEAR_FILENAME)

  if k_fold_split_id is None:
    train_indices = load_npy(data_root / TRAIN_INDEX_FILENAME) #"train_idx.npy"
    valid_indices = load_npy(data_root / VALID_INDEX_FILENAME) #"train_idx.npy"
  else:
    train_indices, valid_indices = get_train_and_valid_idx_for_split(
        k_fold_split_id, num_splits=NUM_K_FOLD_SPLITS,
        root_path=data_root / K_FOLD_SPLITS_DIR)

  array_dict["train_indices"] = train_indices
  array_dict["valid_indices"] = valid_indices
  array_dict["test_indices"] = load_npy(data_root / TEST_INDEX_FILENAME) #"train_idx.npy"

  # if use_fused_node_labels:
  #   array_dict["user_label"] = load_npy(data_root / FUSED_NODE_LABELS_FILENAME) #"fused_node_labels.npy"
  # else:
  #   array_dict["user_label"] = load_npy(data_root / RAW_NODE_LABELS_FILENAME) #"node_label.npy"

  if return_adjacencies:
    logging.info("Starting to get adjacencies.")
    if use_fused_node_adjacencies:
      user_user_index = load_csr(
          data_root / FUSED_USER_EDGES_FILENAME, debug=use_dummy_adjacencies)
      user_user_index_t = load_csr(
          data_root / FUSED_USER_EDGES_T_FILENAME, debug=use_dummy_adjacencies)
    array_dict.update(
        dict(
            # Всевозможные связи между автором и институтами,статьями
            group_institution_index=load_csr(
                data_root / EDGES_GROUP_INSTITUTION, #"user_user_b.npz"
                debug=use_dummy_adjacencies),
            institution_group_index=load_csr(
                data_root / EDGES_INSTITUTION_GROUP, #"institution_group.npz"
                debug=use_dummy_adjacencies)
        ))

  if return_pca_embeddings:
    array_dict["bert_pca_129"] = np.load(
        data_root / PCA_MERGED_FEATURES_FILENAME, mmap_mode="r")
    print(array_dict["bert_pca_129"].shape)
    assert array_dict["bert_pca_129"].shape == (NUM_NODES, 129)

  logging.info("Finish getting files")

  # # pytype: disable=attribute-error
  # assert array_dict["user_year"].shape[0] == NUM_USERS
  # assert array_dict["user_label"].shape[0] == NUM_USERS

  if return_adjacencies and not use_dummy_adjacencies:
    array_dict = _fix_adjacency_shapes(array_dict)

    # Тесты на сохранение количества каждой сущности
    assert array_dict["user_group_index"].shape == (NUM_USERS, NUM_GROUPS)
    assert array_dict["group_user_index"].shape == (NUM_GROUPS, NUM_USERS)
    # assert array_dict["user_user_index"].shape == (NUM_USERS, NUM_USERS)
    # assert array_dict["user_user_index_t"].shape == (NUM_USERS, NUM_USERS)
    # print('array_dict["institution_group_index"].shape',array_dict["institution_group_index"].shape)
    # assert array_dict["institution_group_index"].shape == (
        # NUM_INSTITUTIONS, NUM_GROUPS)
    # assert array_dict["group_institution_index"].shape == (
    #     NUM_GROUPS, NUM_INSTITUTIONS)

  # pytype: enable=attribute-error

  return array_dict


def add_nodes_year(graph, user_year):
  nodes = graph.nodes.copy()
  indices = nodes["index"]
  year = user_year[np.minimum(indices, user_year.shape[0] - 1)].copy()
  year[nodes["type"] != 0] = 1900
  nodes["year"] = year
  return graph._replace(nodes=nodes)


def add_nodes_label(graph, user_label):
  nodes = graph.nodes.copy()
  indices = nodes["index"]
  label = user_label[np.minimum(indices, user_label.shape[0] - 1)]
  label[nodes["type"] != 0] = 0
  nodes["label"] = label
  return graph._replace(nodes=nodes)


def add_nodes_embedding_from_array(graph, array):
  """Adds embeddings from the sstable_service for the indices."""
  nodes = graph.nodes.copy()
  indices = nodes["index"]
  embedding_indices = indices.copy()
  # embedding_indices[nodes["type"] == 1] += NUM_USERS
  # embedding_indices[nodes["type"] == 2] += NUM_USERS + NUM_GROUPS

  # Gather the embeddings for the indices.
  nodes["features"] = array[embedding_indices]
  return graph._replace(nodes=nodes)


def get_graph_subsampling_dataset(
    prefix, arrays, shuffle_indices, ratio_unlabeled_data_to_labeled_data,
    max_nodes, max_edges,
    **subsampler_kwargs):
  """Returns tf_dataset for online sampling."""

  def generator():
    labeled_indices = arrays[f"{prefix}_indices"]
    if ratio_unlabeled_data_to_labeled_data > 0:
      num_unlabeled_data_to_add = int(ratio_unlabeled_data_to_labeled_data *
                                      labeled_indices.shape[0])
      unlabeled_indices = np.random.choice(
          NUM_USERS, size=num_unlabeled_data_to_add, replace=False)
      root_node_indices = np.concatenate([labeled_indices, unlabeled_indices])
    else:
      root_node_indices = labeled_indices
    if shuffle_indices:
      root_node_indices = root_node_indices.copy()
      np.random.shuffle(root_node_indices)

    for index in root_node_indices:
      graph = sub_sampler.subsample_graph(
          index,
          arrays["group_institution_index"],
          arrays["institution_group_index"],
          arrays["group_user_index"],
          arrays["user_group_index"],
          arrays["user_user_index"],
          arrays["user_user_index_t"],
          user_years=arrays["user_year"],
          max_nodes=max_nodes,
          max_edges=max_edges,
          **subsampler_kwargs)

      graph = add_nodes_label(graph, arrays["user_label"])
      graph = add_nodes_year(graph, arrays["user_year"])
      graph = tf_graphs.GraphsTuple(*graph)
      yield graph

  sample_graph = next(generator())

  return tf.data.Dataset.from_generator(
      generator,
      output_signature=utils_tf.specs_from_graphs_tuple(sample_graph))


def user_features_to_group_features(
    group_user_index, user_features):
  """Averages user features to groups."""
  # assert user_features.shape[0] == NUM_USERS
  assert group_user_index.shape[0] == NUM_GROUPS
  group_features = np.zeros(
      [NUM_GROUPS, user_features.shape[1]], dtype=user_features.dtype)
  for group_i in range(NUM_GROUPS):
    user_indices = group_user_index[group_i].indices
    group_features[group_i] = user_features[user_indices].mean(
        axis=0, dtype=np.float32)
    if group_i % 10000 == 0:
      logging.info("%d/%d", group_i, NUM_GROUPS)
  return group_features


def group_features_to_institution_features(
    institution_group_index, group_features):
  """Averages group features to institutions."""
  assert group_features.shape[0] == NUM_GROUPS
  # assert institution_group_index.shape[0] == NUM_INSTITUTIONS
  # institution_features = np.zeros(
  #     [NUM_INSTITUTIONS, group_features.shape[1]], dtype=group_features.dtype)
  # for institution_i in range(NUM_INSTITUTIONS):
  #   group_indices = institution_group_index[institution_i].indices
  #   institution_features[institution_i] = group_features[
  #       group_indices].mean(axis=0, dtype=np.float32)
  #   if institution_i % 10000 == 0:
  #     logging.info("%d/%d", institution_i, NUM_INSTITUTIONS)
  # return institution_features


def generate_fused_user_adjacency_matrix(neighbor_indices, neighbor_distances,
                                          user_user_csr):
  """Generates fused adjacency matrix for identical nodes."""
  # First construct set of identical node indices.
  # NOTE: Since we take only top K=26 identical pairs for each node, this is not
  # actually exhaustive. Also, if A and B are equal, and B and C are equal,
  # this method would not necessarily detect A and C being equal.
  # However, this should capture almost all cases.
  logging.info("Generating fused user adjacency matrix")
  eps = 0.0
  mask = ((neighbor_indices != np.mgrid[:neighbor_indices.shape[0], :1]) &
          (neighbor_distances <= eps))
  identical_pairs = list(map(tuple, np.nonzero(mask)))
  del mask

  # Have a csc version for fast column access.
  user_user_csc = user_user_csr.tocsc()

  # Construct new matrix as coo, starting off with original rows/cols.
  user_user_coo = user_user_csr.tocoo()
  new_rows = [user_user_coo.row]
  new_cols = [user_user_coo.col]

  for pair in tqdm.tqdm(identical_pairs):
    # STEP ONE: First merge users being cited by the pair.
    # Add edges from second user, to all users cited by first user.
    cited_by_first = user_user_csr.getrow(pair[0]).nonzero()[1]
    if cited_by_first.shape[0] > 0:
      new_rows.append(pair[1] * np.ones_like(cited_by_first))
      new_cols.append(cited_by_first)

    # Add edges from first user, to all users cited by second user.
    cited_by_second = user_user_csr.getrow(pair[1]).nonzero()[1]
    if cited_by_second.shape[0] > 0:
      new_rows.append(pair[0] * np.ones_like(cited_by_second))
      new_cols.append(cited_by_second)

    # STEP TWO: Then merge users that cite the pair.
    # Add edges to second user, from all users citing the first user.
    citing_first = user_user_csc.getcol(pair[0]).nonzero()[0]
    if citing_first.shape[0] > 0:
      new_rows.append(citing_first)
      new_cols.append(pair[1] * np.ones_like(citing_first))

    # Add edges to first user, from all users citing the second user.
    citing_second = user_user_csc.getcol(pair[1]).nonzero()[0]
    if citing_second.shape[0] > 0:
      new_rows.append(citing_second)
      new_cols.append(pair[0] * np.ones_like(citing_second))

  logging.info("Done with adjacency loop")
  user_user_coo_shape = user_user_coo.shape
  del user_user_csr
  del user_user_csc
  del user_user_coo
  # All done; now concatenate everything together and form new matrix.
  new_rows = np.concatenate(new_rows)
  new_cols = np.concatenate(new_cols)
  return sp.coo_matrix(
      (np.ones_like(new_rows, dtype=bool), (new_rows, new_cols)),
      shape=user_user_coo_shape).tocsr()

# данные -> train, valid k раз
def generate_k_fold_splits(
    train_idx, valid_idx, output_path, num_splits=NUM_K_FOLD_SPLITS):
  """Generates splits adding fractions of the validation split to training."""
  output_path = Path(output_path)
  np.random.seed(42)
  valid_idx = np.random.permutation(valid_idx)
  # Split into `num_parts` (almost) identically sized arrays.
  valid_idx_parts = np.array_split(valid_idx, num_splits)

  for i in range(num_splits):
    # Add all but the i'th subpart to training set.
    new_train_idx = np.concatenate(
        [train_idx, *valid_idx_parts[:i], *valid_idx_parts[i+1:]])
    # i'th subpart is validation set.
    new_valid_idx = valid_idx_parts[i]
    train_path = output_path / f"train_idx_{i}_{num_splits}.npy"
    valid_path = output_path / f"valid_idx_{i}_{num_splits}.npy"
    np.save(train_path, new_train_idx)
    np.save(valid_path, new_valid_idx)
    logging.info("Saved: %s", train_path)
    logging.info("Saved: %s", valid_path)


def get_train_and_valid_idx_for_split(
    split_id: int,
    num_splits: int,
    root_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns train and valid indices for given split."""
  new_train_idx = load_npy(f"{root_path}/train_idx_{split_id}_{num_splits}.npy")
  new_valid_idx = load_npy(f"{root_path}/valid_idx_{split_id}_{num_splits}.npy")
  return new_train_idx, new_valid_idx

# (?)
def generate_fused_node_labels(neighbor_indices, neighbor_distances,
                               node_labels, train_indices, valid_indices,
                               test_indices):
  """Generates fused adjacency matrix for identical nodes."""
  logging.info("Generating fused node labels")
  valid_indices = set(valid_indices.tolist())
  test_indices = set(test_indices.tolist())
  valid_or_test_indices = valid_indices | test_indices

  train_indices = train_indices[train_indices < neighbor_indices.shape[0]]
  # Go through list of all pairs where one node is in training set, and
  for i in tqdm.tqdm(train_indices):
    for j in range(neighbor_indices.shape[1]):
      other_index = neighbor_indices[i][j]
      # if the other is not a validation or test node,
      if other_index in valid_or_test_indices:
        continue
      # and they are identical,
      if neighbor_distances[i][j] == 0:
        # assign the label of the training node to the other node
        node_labels[other_index] = node_labels[i]

  return node_labels


def _pad_to_shape(
    sparse_csr_matrix: sp.csr_matrix,
    output_shape: Tuple[int, int]) -> sp.csr_matrix:
  """Pads a csr sparse matrix to the given shape."""

  # We should not try to expand anything smaller.
  print('output_shape', output_shape)
  print('sparse_csr_matrix.shape', sparse_csr_matrix.shape)
  assert np.all(sparse_csr_matrix.shape <= output_shape)

  # Maybe it already has the right shape.
  if sparse_csr_matrix.shape == output_shape:
    return sparse_csr_matrix

  # Append as many indptr elements as we need to match the leading size,
  # This is achieved by just padding with copies of the last indptr element.
  required_padding = output_shape[0] - sparse_csr_matrix.shape[0]
  updated_indptr = np.concatenate(
      [sparse_csr_matrix.indptr] +
      [sparse_csr_matrix.indptr[-1:]] * required_padding,
      axis=0)

  # The change in trailing size does not have structural implications, it just
  # determines the highest possible value for the indices, so it is sufficient
  # to just pass the new output shape, with the correct trailing size.
  return sp.csr.csr_matrix(
      (sparse_csr_matrix.data,
       sparse_csr_matrix.indices,
       updated_indptr),
      shape=output_shape)


def _fix_adjacency_shapes(
    arrays: Dict[str, sp.csr.csr_matrix],
    ) -> Dict[str, sp.csr.csr_matrix]:
  """Fixes the shapes of the adjacency matrices."""
  arrays = arrays.copy()
  for key in [
              # "group_institution_index",
              "group_user_index",
              "user_user_index",
              # "institution_group_index",
              "user_group_index",
              "user_user_index_t"
              ]:
    print('key', key)
    type_sender = key.split("_")[0]
    print('type_sender', type_sender)
    print('SIZES[type_sender]', SIZES[type_sender])
    type_receiver = key.split("_")[1]
    print('type_receiver', type_receiver)
    print('SIZES[type_receiver]', SIZES[type_receiver])
    arrays[key] = _pad_to_shape(
        arrays[key], output_shape=(SIZES[type_sender], SIZES[type_receiver]))
  return arrays
