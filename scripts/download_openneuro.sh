#!/usr/bin/env bash
set -euo pipefail

# OpenNeuro dataset and version used by this pipeline
DATASET_ID="ds008083"
DATASET_VERSION="1.0.0"

# Find the repository root:
# this script lives in scripts/, so the repo root is one folder above it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Download location: same repository as the pipeline
DATA_ROOT="${REPO_ROOT}/data"
DATASET_DIR="${DATA_ROOT}/BIDShierPriors"

echo "Repository root: ${REPO_ROOT}"
echo "Dataset: ${DATASET_ID}"
echo "Version: ${DATASET_VERSION}"
echo "Target directory: ${DATASET_DIR}"

mkdir -p "${DATA_ROOT}"

# Check that OpenNeuro CLI exists
if ! command -v openneuro >/dev/null 2>&1; then
  echo "ERROR: OpenNeuro CLI is not installed or not on PATH."
  echo ""
  echo "Install it with:"
  echo "  deno install -A --global jsr:@openneuro/cli -n openneuro"
  echo ""
  echo "Then make sure ~/.deno/bin is on your PATH."
  exit 1
fi

# Download or update the OpenNeuro dataset
if [ -d "${DATASET_DIR}/.git" ]; then
  echo "Existing OpenNeuro clone found; updating to version ${DATASET_VERSION}..."
  openneuro download "${DATASET_ID}" --version "${DATASET_VERSION}" "${DATASET_DIR}"
elif [ -d "${DATASET_DIR}" ]; then
  echo "ERROR: ${DATASET_DIR} exists but is not an OpenNeuro git clone."
  echo "Remove it and rerun, e.g.:"
  echo "  rm -rf ${DATASET_DIR}"
  exit 1
else
  echo "Downloading ${DATASET_ID} (version ${DATASET_VERSION}) from OpenNeuro..."
  openneuro download "${DATASET_ID}" --version "${DATASET_VERSION}" "${DATASET_DIR}"
fi

cd "${DATASET_DIR}"

# Retrieve annexed file contents if DataLad or git-annex is available
if command -v datalad >/dev/null 2>&1; then
  echo "Retrieving full dataset contents with DataLad..."
  datalad get .
elif command -v git-annex >/dev/null 2>&1; then
  echo "Retrieving full dataset contents with git-annex..."
  git-annex get .
else
  echo "WARNING: Neither datalad nor git-annex was found."
  echo "The dataset structure may be present, but large file contents may not be downloaded."
  echo ""
  echo "Install DataLad or git-annex, then run:"
  echo "  cd ${DATASET_DIR}"
  echo "  datalad get ."
  echo "or:"
  echo "  git-annex get ."
fi

echo ""
echo "Done."
echo "Dataset is available at:"
echo "  ${DATASET_DIR}"