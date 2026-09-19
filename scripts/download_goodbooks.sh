#!/bin/bash
# Download and extract the GoodBooks-10k dataset used by the Session 02 TP.
# The archive lands in <repo root>/data/goodbooks-10k (ignored by git).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/goodbooks-10k"
URL="https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip"

if [ -f "${DATA_DIR}/ratings.csv" ]; then
    echo "Data already downloaded in ${DATA_DIR}"
    exit 0
fi

mkdir -p "${DATA_DIR}"
ARCHIVE="${REPO_ROOT}/data/goodbooks-10k.zip"

echo "Downloading ${URL}"
if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "${ARCHIVE}" "${URL}"
else
    wget -O "${ARCHIVE}" "${URL}"
fi

echo "Extracting into ${DATA_DIR}"
if command -v unzip >/dev/null 2>&1; then
    unzip -o -q "${ARCHIVE}" -d "${DATA_DIR}"
else
    python3 -c "import zipfile; zipfile.ZipFile('${ARCHIVE}').extractall('${DATA_DIR}')"
fi

echo "Done:"
ls -1 "${DATA_DIR}"
