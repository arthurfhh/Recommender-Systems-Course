#!/bin/bash
# Download and extract the pre-reduced MovieLens subset used by the Session 03 TP
# (443 users, 4778 movies, plus the pre-computed X, W, b used to check the cost).
#
# Usage: bash scripts/download_movielens_small.sh [destination directory]
# The default destination is src/03_CollaborativeFiltering/data (ignored by git).
#
# The archive is hosted on MEGA. `megadl` (megatools) is used when it is
# installed; otherwise the public link is resolved through the MEGA API with
# curl and the archive is decrypted with openssl (AES-128-CTR), so that the TP
# also runs on machines where megatools cannot be installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${REPO_ROOT}/src/03_CollaborativeFiltering/data}"
FILE_ID="hEJhURAZ"
FILE_KEY="8VeFh14WBrNnMvZxrOfrcXPpHwQDIssXJn8KTYFrPiQ"
URL="https://mega.nz/file/${FILE_ID}#${FILE_KEY}"

if [ -f "${DATA_DIR}/small_movies_Y.csv" ]; then
    echo "Data already downloaded in ${DATA_DIR}"
    exit 0
fi

mkdir -p "${DATA_DIR}"
ARCHIVE="${DATA_DIR}/movielens_small.zip"

if command -v python3 >/dev/null 2>&1 && python3 -c "" >/dev/null 2>&1; then
    PYTHON=python3
else
    PYTHON=python
fi

echo "Downloading ${URL}"
if command -v megadl >/dev/null 2>&1; then
    megadl --path "${ARCHIVE}" "${URL}"
else
    # The 256-bit key of the link folds into the AES key (first half XOR second
    # half) and the CTR nonce (third quarter, the counter starting at zero).
    read -r AES_KEY AES_IV < <(("${PYTHON}" - "${FILE_KEY}" <<'EOF'
import base64, sys

raw = base64.urlsafe_b64decode(sys.argv[1] + "=" * (-len(sys.argv[1]) % 4))
key = bytes(a ^ b for a, b in zip(raw[:16], raw[16:]))
print(key.hex(), raw[16:24].hex() + "00" * 8)
EOF
    ) | tr -d '\015')
    DOWNLOAD_URL="$(curl -s --fail -X POST "https://g.api.mega.co.nz/cs?id=1" \
        -d "[{\"a\":\"g\",\"g\":1,\"p\":\"${FILE_ID}\"}]" \
        | "${PYTHON}" -c "import json, sys; print(json.load(sys.stdin)[0]['g'])" | tr -d '\015')"
    curl -L --fail -o "${ARCHIVE}.enc" "${DOWNLOAD_URL}"
    openssl enc -d -aes-128-ctr -K "${AES_KEY}" -iv "${AES_IV}" \
        -in "${ARCHIVE}.enc" -out "${ARCHIVE}"
    rm -f "${ARCHIVE}.enc"
fi

echo "Extracting into ${DATA_DIR}"
"${PYTHON}" - "${ARCHIVE}" "${DATA_DIR}" <<'EOF'
import sys, zipfile
from pathlib import Path

archive, destination = zipfile.ZipFile(sys.argv[1]), Path(sys.argv[2])
# Flatten the archive: the csv files land directly in the destination.
for member in archive.infolist():
    if member.filename.endswith(".csv") and "__MACOSX" not in member.filename:
        (destination / Path(member.filename).name).write_bytes(archive.read(member))
EOF
rm -f "${ARCHIVE}"

echo "Done:"
ls -1 "${DATA_DIR}"
