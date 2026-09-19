#!/bin/bash
# Download the MovieLens data used by Session 01 (Introduction).
#
#   src/data/ml-100k/                      ml-100k archive (100 000 ratings)
#   src/data/movielens_complete/ml-10m/    ml-10m ratings/movies, as csv
#
# The canonical source is GroupLens:
#
#   wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
#   wget http://files.grouplens.org/datasets/movielens/ml-10m.zip
#
# Some networks block files.grouplens.org; the script then falls back to a
# GitHub mirror of the same two archives. Everything lands in src/data, which
# is git-ignored.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${REPO_ROOT}/src/data"
GROUPLENS="http://files.grouplens.org/datasets/movielens"
MIRROR="https://raw.githubusercontent.com/Blosc/movielens-bench/master"
TIMEOUT=120

mkdir -p "${DATA}/movielens_complete"

fetch() { # fetch <url> <destination>
    curl -fsSL --max-time "${TIMEOUT}" -o "$2" "$1"
}

# --------------------------------------------------------------------------- #
# MovieLens 100K — the dataset used from the second half of the notebook on
# --------------------------------------------------------------------------- #
if [ -f "${DATA}/ml-100k/u.item" ]; then
    echo "ml-100k already downloaded"
elif fetch "${GROUPLENS}/ml-100k.zip" "${DATA}/ml-100k.zip"; then
    unzip -o -q "${DATA}/ml-100k.zip" -d "${DATA}"
    rm -f "${DATA}/ml-100k.zip"
    echo "ml-100k downloaded from GroupLens"
else
    echo "GroupLens unreachable, falling back to the GitHub mirror for ml-100k"
    mkdir -p "${DATA}/ml-100k"
    for file in u.data u.item u.user u.genre u.info; do
        fetch "${MIRROR}/ml-100k/${file}" "${DATA}/ml-100k/${file}" || {
            echo "could not download ${file}" >&2
            exit 1
        }
    done
fi

# --------------------------------------------------------------------------- #
# MovieLens 10M — the "too big to handle comfortably" dataset
# --------------------------------------------------------------------------- #
BIG="${DATA}/movielens_complete/ml-10m"
if [ -f "${BIG}/ratings.csv" ]; then
    echo "ml-10m already downloaded"
else
    mkdir -p "${BIG}"
    RAW_RATINGS=""
    RAW_MOVIES=""
    if fetch "${GROUPLENS}/ml-10m.zip" "${DATA}/ml-10m.zip"; then
        unzip -o -q "${DATA}/ml-10m.zip" -d "${DATA}"
        rm -f "${DATA}/ml-10m.zip"
        RAW_RATINGS="${DATA}/ml-10M100K/ratings.dat"
        RAW_MOVIES="${DATA}/ml-10M100K/movies.dat"
        echo "ml-10m downloaded from GroupLens"
    else
        echo "GroupLens unreachable, falling back to the GitHub mirror for ml-10m"
        mkdir -p "${DATA}/ml-10m-raw"
        fetch "${MIRROR}/ml-10m/ratings.dat.gz" "${DATA}/ml-10m-raw/ratings.dat.gz" &&
            fetch "${MIRROR}/ml-10m/movies.dat" "${DATA}/ml-10m-raw/movies.dat" || {
            echo "could not download ml-10m" >&2
            exit 1
        }
        RAW_RATINGS="${DATA}/ml-10m-raw/ratings.dat.gz"
        RAW_MOVIES="${DATA}/ml-10m-raw/movies.dat"
    fi

    # The .dat files use "::" (GroupLens) or ";" (mirror) as separator: convert
    # them to the plain csv layout of the recent MovieLens releases.
    python3 - "${RAW_RATINGS}" "${RAW_MOVIES}" "${BIG}" <<'PY'
import csv
import gzip
import sys

ratings_path, movies_path, out_dir = sys.argv[1:4]


def rows(path, encoding="utf-8"):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding=encoding, errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            yield line.split("::") if "::" in line else line.split(";")


with open(f"{out_dir}/ratings.csv", "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["userId", "movieId", "rating", "timestamp"])
    n = 0
    for fields in rows(ratings_path):
        writer.writerow(fields[:4])
        n += 1
print(f"{n:,} ratings written to {out_dir}/ratings.csv")

with open(f"{out_dir}/movies.csv", "w", newline="", encoding="ISO-8859-1") as out:
    writer = csv.writer(out)
    writer.writerow(["movieId", "title", "genres"])
    m = 0
    for fields in rows(movies_path, encoding="ISO-8859-1"):
        writer.writerow([fields[0], "::".join(fields[1:-1]), fields[-1]])
        m += 1
print(f"{m:,} movies written to {out_dir}/movies.csv")
PY
fi

echo
echo "Data ready:"
du -sh "${DATA}/ml-100k" "${BIG}"
