#!/usr/bin/env bash
set -euo pipefail

# build_leanstore.sh - configure and build the leanstore submodule
# Usage:
#   ./build_leanstore.sh        # configure + build (default)
#   ./build_leanstore.sh --configure-only
#   ./build_leanstore.sh --help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEANSTORE_DIR="${SCRIPT_DIR}/leanstore"
BUILD_DIR="${LEANSTORE_DIR}/build"

CONFIGURE_ONLY=0
INSTALL_DEPS=0
ASSUME_YES=0

if [[ ${#} -gt 0 ]]; then
  case "$1" in
    --configure-only)
      CONFIGURE_ONLY=1
      ;;
    --install-deps)
      INSTALL_DEPS=1
      ;;
    --yes|-y)
      ASSUME_YES=1
      ;;
    --help|-h)
      echo "Usage: $0 [--configure-only] [--install-deps] [--yes]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--configure-only]"
      exit 1
      ;;
  esac
fi

if [ ! -d "${LEANSTORE_DIR}" ]; then
  echo "Error: leanstore submodule not found at ${LEANSTORE_DIR}"
  exit 1
fi

if [ "${INSTALL_DEPS}" -eq 1 ]; then
  echo "Installing LeanStore build dependencies via apt (requires sudo)..."
  PKGS=(cmake libtbb2-dev libaio-dev libsnappy-dev zlib1g-dev libbz2-dev liblz4-dev libzstd-dev librocksdb-dev liblmdb-dev libwiredtiger-dev liburing-dev)
  echo "Packages: ${PKGS[*]}"
  if [ "${ASSUME_YES}" -eq 0 ]; then
    read -p "Proceed with apt-get install? [y/N] " ans || true
  else
    ans=y
  fi
  case "${ans}" in
    [Yy]*)
      sudo apt-get update
      sudo apt-get install -y "${PKGS[@]}"
      ;;
    *)
      echo "Skipping dependency installation." ;;
  esac
fi

echo "Building leanstore in: ${LEANSTORE_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "Running cmake (RelWithDebInfo)..."
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

if [ "${CONFIGURE_ONLY}" -eq 1 ]; then
  echo "Configuration finished. Skipping build as requested."
  exit 0
fi

echo "Starting build (make -j$(nproc))..."
make -j"$(nproc)"

echo "Build complete. Built artifacts are in: ${BUILD_DIR}"