#!/bin/bash
set -e
set -x


while getopts ":r:" opt; do
  case ${opt} in
    r )
      TASK_ROOT=$OPTARG
      ;;
    \? )
      echo "Usage: preprocess_data.sh -r <Task root directory>"
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))

# Get this script's directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


DATA_ROOT="${TASK_ROOT}"/data
PREPROCESSED_DIR="${DATA_ROOT}"/preprocessed

# Create preprocessed directory to move all files to it.
mkdir -p "${PREPROCESSED_DIR}"

# Run the CSR edge builder.
python "${SCRIPT_DIR}"/csr_builder.py --data_root="${DATA_ROOT}"

# Run the PCA feature builder.
python "${SCRIPT_DIR}"/pca_builder.py --data_root="${DATA_ROOT}"

# Run the neighbor-finder/fuser builder.
python "${SCRIPT_DIR}"/neighbor_builder.py --data_root="${DATA_ROOT}"

# Run the validation split generator.
python "${SCRIPT_DIR}"/generate_validation_splits.py \
  --data_root="${DATA_ROOT}" \
  --output_dir="${DATA_ROOT}/k_fold_splits"
