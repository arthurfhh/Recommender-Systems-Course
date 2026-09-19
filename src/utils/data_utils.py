"""Loading helpers for the MovieLens 100K dataset used in Session 01.

The files are read from a **local copy** of the `ml-100k` archive when there is
one (`src/data/ml-100k` or `<repo>/data/ml-100k`), and from the GroupLens server
otherwise. The local copy is what makes the notebook runnable offline, or behind
a network policy that blocks `files.grouplens.org`; `scripts/download_movielens.sh`
creates it.

`load_data()` keeps the signature it has always had, so the notebooks that call
it do not change.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROUPLENS_URL = "http://files.grouplens.org/datasets/movielens/ml-100k"

#: Directories searched for a local copy of the ml-100k archive, in order.
LOCAL_DIRS = (
    Path(__file__).resolve().parents[1] / "data" / "ml-100k",  # src/data/ml-100k
    Path(__file__).resolve().parents[2] / "data" / "ml-100k",  # <repo>/data/ml-100k
)

RATING_COLUMNS = ["UserId", "MovieId", "Rating", "Timestamp"]
USER_COLUMNS = ["UserId", "Age", "Gender", "Occupation", "ZipCode"]
ITEM_COLUMNS = [
    "MovieId",
    "Title",
    "Date",
    "VideoReleaseDate",
    "Url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def local_dir() -> Path:
    """Return the local ml-100k directory, or `None` when there is none."""
    for candidate in LOCAL_DIRS:
        if (candidate / "u.item").exists():
            return candidate
    return None


def data_source(filename: str) -> str:
    """Path of a ml-100k file: the local copy if it exists, the URL otherwise.

    Parameters
    ----------
    filename : str
        Name of the file inside the archive, e.g. ``"u.item"``.

    Returns
    -------
    str
        A local path or a `http://files.grouplens.org/...` URL, either of which
        `pandas.read_csv` can read.
    """
    directory = local_dir()
    if directory is not None and (directory / filename).exists():
        return str(directory / filename)
    return f"{GROUPLENS_URL}/{filename}"


def load_ratings(filename: str = "u.data") -> pd.DataFrame:
    """Load a ml-100k rating file (`u.data`, `u1.base`, `u1.test`, ...)."""
    ratings = pd.read_csv(data_source(filename), sep="\t", engine="python", header=None)
    ratings.columns = RATING_COLUMNS
    return ratings


def load_users() -> pd.DataFrame:
    """Load the demographic table, indexed by `UserId`."""
    users = pd.read_csv(data_source("u.user"), sep="|", engine="python", header=None)
    users.columns = USER_COLUMNS
    return users.set_index("UserId")


def load_items() -> pd.DataFrame:
    """Load the movie table (title, date and multi-hot genres), indexed by `MovieId`."""
    items = pd.read_csv(
        data_source("u.item"),
        sep="|",
        engine="python",
        encoding="ISO-8859-1",
        header=None,
    )
    items.columns = ITEM_COLUMNS
    return items.set_index("MovieId")


def split_ratings(
    ratings: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible train/test split, used when `u1.base`/`u1.test` are missing.

    GroupLens ships five ready-made 80/20 splits (`u1` to `u5`); some mirrors of
    the archive only carry `u.data`, so we rebuild an equivalent split here.
    """
    generator = np.random.default_rng(seed)
    shuffled = ratings.iloc[generator.permutation(len(ratings))]
    n_test = int(round(test_size * len(ratings)))
    test = shuffled.iloc[:n_test].sort_index().reset_index(drop=True)
    train = shuffled.iloc[n_test:].sort_index().reset_index(drop=True)
    return train, test


def load_train_test() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the (train, test) rating tables of the ml-100k dataset."""
    directory = local_dir()
    has_official_split = directory is None or (directory / "u1.base").exists()
    if has_official_split:
        return load_ratings("u1.base"), load_ratings("u1.test")

    logger.info("u1.base/u1.test not found locally: splitting u.data instead")
    return split_ratings(load_ratings("u.data"))


def load_data():
    """Load everything the Session 01 notebook needs.

    Returns
    -------
    tuple
        ``(df_rating, df_rating_test, df_users, df_items, df_matrix, n_users,
        n_items)`` — the rating table and its test counterpart, the users, the
        movies, the (movies x users) rating matrix and the two cardinalities.
    """
    df_rating, df_rating_test = load_train_test()
    df_users = load_users()
    df_items = load_items()

    # Pivot rating table in order to get rating matrix
    df_matrix = df_rating.pivot(index="MovieId", columns="UserId", values="Rating")

    n_users = len(df_users)
    n_items = len(df_items)

    return df_rating, df_rating_test, df_users, df_items, df_matrix, n_users, n_items
