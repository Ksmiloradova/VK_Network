

set -e

while getopts ":i:o:" opt; do
  case ${opt} in
    i )
      INPUT_DIR=$OPTARG
      ;;
    o )
      TASK_ROOT=$OPTARG
      ;;
    \? )
      echo "Usage: organize_data.sh -i <Downloaded data dir> -o <Task root directory>"
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Get this script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [[ -z "${INPUT_DIR}" ]]; then
  echo "Need INPUT_DIR argument (-i <INPUT_DIR>)"
  exit 1
fi

if [[ -z "${TASK_ROOT}" ]]; then
  echo "Need TASK_ROOT argument (-o <TASK_ROOT>)"
  exit 1
fi

DATA_ROOT="${TASK_ROOT}"/data
# 
# Create raw directory to move all files to it.
# mkdir "${INPUT_DIR}"/mag240m_kddcup2021/raw
# 
# mv "${INPUT_DIR}"/mag240m_kddcup2021/processed/user/node_feat.npy \
  #  "${INPUT_DIR}"/mag240m_kddcup2021/processed/user/node_label.npy \
  #  "${INPUT_DIR}"/mag240m_kddcup2021/processed/user/node_year.npy \
  #  "${DATA_ROOT}"/raw
# mv "${INPUT_DIR}"/mag240m_kddcup2021/processed/group___affiliated_with___institution/edge_index.npy \
  #  "${DATA_ROOT}"/raw/group_affiliated_with_institution_edges.npy
# mv "${ROOT}"/mag240m_kddcup2021/processed/group___writes___user/edge_index.npy \
  #  "${DATA_ROOT}"/raw/group_writes_user_edges.npy
# mv "${ROOT}"/mag240m_kddcup2021/processed/user___cites___user/edge_index.npy \
  #  "${DATA_ROOT}"/raw/user_cites_user_edges.npy

# Split and save the train/valid/test indices to the raw directory, with names
# "train_idx.npy", "valid_idx.npy", "test_idx.npy":
python3 "${SCRIPT_DIR}"/split_and_save_indices.py --data_root=${DATA_ROOT}
