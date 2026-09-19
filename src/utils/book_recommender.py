"""Utilities for the GoodBooks-10k recommender exercise (Session 02 TP).

The module collects everything the exercise notebook needs:

* loading and cleaning the `goodbooks-10k` dataset,
* building a textual content representation of a book (title, authors, tags),
* splitting the ratings into a train/test set *per user*,
* three recommenders (popularity baseline, biased matrix factorisation,
  content-based TF-IDF) plus a hybrid of the last two,
* the metrics used to compare them (RMSE/MAE and Precision/Recall/NDCG@k).

Only `numpy`, `pandas` and `scikit-learn` are required, so the code runs on the
course environment without any extra dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

__all__ = [
    "load_goodbooks",
    "build_content_corpus",
    "train_test_split_by_user",
    "group_items_by_user",
    "IdIndex",
    "StaticRanker",
    "PopularityRecommender",
    "BiasedMF",
    "ItemKNNRecommender",
    "ContentBasedRecommender",
    "HybridRecommender",
    "rmse",
    "mae",
    "precision_recall_at_k",
    "ndcg_at_k",
    "evaluate_ranking",
    "evaluate_cold_start",
    "top_n_recommendations",
]

# Tags that carry shelving information instead of content information.
SHELF_TAGS = frozenset(
    {
        "to-read",
        "currently-reading",
        "books-i-own",
        "owned",
        "owned-books",
        "i-own",
        "favorites",
        "favourites",
        "my-books",
        "my-library",
        "default",
        "ebook",
        "ebooks",
        "kindle",
        "audiobook",
        "audiobooks",
        "library",
        "to-buy",
        "book-club",
        "re-read",
        "reread",
        "read-in-2016",
        "read-in-2015",
        "books",
        "all-time-favorites",
        "series",
    }
)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_goodbooks(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load the GoodBooks-10k csv files.

    Parameters
    ----------
    data_dir : str or Path
        Directory holding `books.csv`, `ratings.csv`, `book_tags.csv` and
        `tags.csv` (as extracted from the goodbooks-10k archive).

    Returns
    -------
    dict of str -> pandas.DataFrame
        Keys: ``books``, ``ratings``, ``book_tags``, ``tags``.
    """
    data_dir = Path(data_dir)
    missing = [
        name
        for name in ("books.csv", "ratings.csv", "book_tags.csv", "tags.csv")
        if not (data_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {data_dir}. Run scripts/download_goodbooks.sh first."
        )

    return {
        "books": pd.read_csv(data_dir / "books.csv"),
        "ratings": pd.read_csv(data_dir / "ratings.csv"),
        "book_tags": pd.read_csv(data_dir / "book_tags.csv"),
        "tags": pd.read_csv(data_dir / "tags.csv"),
    }


def build_content_corpus(
    books: pd.DataFrame,
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
    n_tags: int = 12,
    drop_shelf_tags: bool = True,
    use_authors: bool = True,
    use_tags: bool = True,
) -> pd.Series:
    """Build one text document per book out of its title, authors and tags.

    Authors are turned into single tokens (``suzanne_collins``) so that the
    vectoriser cannot match two different people sharing a first name.

    Parameters
    ----------
    books, book_tags, tags : pandas.DataFrame
        The corresponding GoodBooks-10k tables.
    n_tags : int
        Number of most frequently applied tags kept per book.
    drop_shelf_tags : bool
        Whether to drop shelving tags (``to-read``, ``owned``, ...) which say
        nothing about the content of the book.
    use_authors, use_tags : bool
        Switches used by the feature ablation of the exercise notebook.

    Returns
    -------
    pandas.Series
        Indexed by ``book_id``, one text document per book.
    """
    if not use_tags:
        top_tags = pd.Series(dtype=str)
    else:
        top_tags = _top_tags_per_book(book_tags, tags, books, n_tags, drop_shelf_tags)

    authors = (
        books.authors.fillna("")
        .str.lower()
        .str.replace(r"[^a-z, ]", "", regex=True)
        .str.replace(", ", ",", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace(",", " ", regex=False)
        if use_authors
        else pd.Series("", index=books.index)
    )

    corpus = (
        books.title.fillna("")
        + " "
        + books.original_title.fillna("")
        + " "
        + authors
        + " "
        + books.book_id.map(top_tags).fillna("")
    ).str.strip()
    corpus.index = books.book_id.values
    return corpus.rename("content")


def _top_tags_per_book(
    book_tags: pd.DataFrame,
    tags: pd.DataFrame,
    books: pd.DataFrame,
    n_tags: int,
    drop_shelf_tags: bool,
) -> pd.Series:
    """Most frequently applied tags of each book, as a single string."""
    tagged = (
        book_tags.merge(tags, on="tag_id", how="inner")
        .merge(books[["book_id", "goodreads_book_id"]], on="goodreads_book_id")
        .assign(
            tag_name=lambda df: df.tag_name.str.lower().str.replace(
                r"[^a-z0-9]+", "-", regex=True
            )
        )
    )
    tagged = tagged[tagged.tag_name.str.len() > 2]
    if drop_shelf_tags:
        tagged = tagged[~tagged.tag_name.isin(SHELF_TAGS)]

    return (
        tagged.sort_values("count", ascending=False)
        .groupby("book_id")
        .head(n_tags)
        .groupby("book_id")
        .tag_name.apply(" ".join)
    )


# --------------------------------------------------------------------------- #
# Train / test split
# --------------------------------------------------------------------------- #
def train_test_split_by_user(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    min_train_ratings: int = 3,
    seed: int = 42,
    user_col: str = "user_id",
    item_col: str = "book_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out a fraction of *each* user's ratings.

    A plain random split would put some users entirely in the test set, where no
    personalised model can say anything about them. Splitting inside each user's
    history keeps every test user known at training time, which is what we want
    when we measure the accuracy of a personalised model.

    Items or users that would appear only in the test set are dropped from it.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Must contain `user_col`, `item_col` and a ``rating`` column.
    test_size : float
        Fraction of each user's ratings moved to the test set.
    min_train_ratings : int
        Users are guaranteed to keep at least this many ratings in the train set
        (users with fewer ratings than that keep all of them).
    seed : int
        Seed of the random shuffle.

    Returns
    -------
    (train, test) : tuple of pandas.DataFrame
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must lie strictly between 0 and 1")

    rng = np.random.default_rng(seed)
    shuffled = ratings.sample(frac=1.0, random_state=rng.integers(1 << 31)).reset_index(
        drop=True
    )

    position = shuffled.groupby(user_col).cumcount()
    n_ratings = shuffled.groupby(user_col)[user_col].transform("size")
    n_test = np.minimum(
        np.floor(n_ratings * test_size), np.maximum(n_ratings - min_train_ratings, 0)
    )

    is_test = position < n_test
    train = shuffled[~is_test].reset_index(drop=True)
    test = shuffled[is_test]

    known_users = set(train[user_col].unique())
    known_items = set(train[item_col].unique())
    test = test[
        test[user_col].isin(known_users) & test[item_col].isin(known_items)
    ].reset_index(drop=True)
    return train, test


def group_items_by_user(
    users: np.ndarray,
    items: np.ndarray,
    n_users: int,
    mask: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """Group encoded item indices by encoded user index.

    Used to build, for every user, the set of items already seen in training
    (to be masked at recommendation time) and the set of relevant test items.
    """
    users, items = np.asarray(users), np.asarray(items)
    if mask is not None:
        users, items = users[mask], items[mask]
    order = np.argsort(users, kind="stable")
    users, items = users[order], items[order]
    bounds = np.searchsorted(users, np.arange(n_users + 1))
    return {
        user: items[bounds[user] : bounds[user + 1]]
        for user in range(n_users)
        if bounds[user + 1] > bounds[user]
    }


# --------------------------------------------------------------------------- #
# Index helper
# --------------------------------------------------------------------------- #
@dataclass
class IdIndex:
    """Maps raw ids to the contiguous integer indices the models work with."""

    users: np.ndarray
    items: np.ndarray
    user_pos: Dict[int, int] = field(init=False, repr=False)
    item_pos: Dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.user_pos = {u: k for k, u in enumerate(self.users)}
        self.item_pos = {i: k for k, i in enumerate(self.items)}

    @classmethod
    def from_ratings(
        cls, ratings: pd.DataFrame, user_col: str = "user_id", item_col: str = "book_id"
    ) -> "IdIndex":
        return cls(
            users=np.sort(ratings[user_col].unique()),
            items=np.sort(ratings[item_col].unique()),
        )

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def encode_users(self, ids: Iterable[int]) -> np.ndarray:
        return np.fromiter((self.user_pos[u] for u in ids), dtype=np.int64)

    def encode_items(self, ids: Iterable[int]) -> np.ndarray:
        return np.fromiter((self.item_pos[i] for i in ids), dtype=np.int64)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class StaticRanker:
    """Wraps a fixed per-item score vector so it can be evaluated like a model."""

    def __init__(self, scores: np.ndarray) -> None:
        self.scores = np.asarray(scores, dtype=float)

    def score_all_items(self, user: int) -> np.ndarray:
        """Return the same score vector whatever the user (non-personalised)."""
        return self.scores


class PopularityRecommender:
    """Non-personalised baseline: damped item mean rating.

    The score of an item is a Bayesian (shrunk) average

    .. math:: \\hat r_i = \\frac{n_i \\bar r_i + m \\mu}{n_i + m}

    which pulls items with few ratings towards the global mean ``mu``. Without
    the damping term the top of the ranking is filled with books rated 5 by a
    single user.
    """

    def __init__(self, damping: float = 25.0) -> None:
        self.damping = damping

    def fit(
        self, train: pd.DataFrame, item_col: str = "book_id"
    ) -> "PopularityRecommender":
        grouped = train.groupby(item_col).rating.agg(["count", "mean"])
        self.global_mean_ = float(train.rating.mean())
        self.scores_ = (
            grouped["count"] * grouped["mean"] + self.damping * self.global_mean_
        ) / (grouped["count"] + self.damping)
        self.counts_ = grouped["count"]
        return self

    def predict(self, users: Sequence[int], items: Sequence[int]) -> np.ndarray:
        """Predicted rating (the user is ignored: the model is not personalised)."""
        return self.scores_.reindex(items).fillna(self.global_mean_).to_numpy()

    def ranker(self, item_ids: Sequence[int], by: str = "count") -> "StaticRanker":
        """Turn the fitted model into a top-N ranker over `item_ids`.

        ``by="count"`` ranks by number of ratings (the *MostPopular* baseline),
        ``by="score"`` ranks by damped mean rating (the *BestRated* baseline).
        """
        if by == "count":
            scores = self.counts_.reindex(item_ids).fillna(0.0)
        elif by == "score":
            scores = self.scores_.reindex(item_ids).fillna(self.global_mean_)
        else:
            raise ValueError("by must be 'count' or 'score'")
        return StaticRanker(scores.to_numpy())

    def top_n(self, n: int = 10, exclude: Optional[Iterable[int]] = None) -> pd.Series:
        scores = self.scores_
        if exclude is not None:
            scores = scores.drop(index=list(exclude), errors="ignore")
        return scores.sort_values(ascending=False).head(n)


class BiasedMF:
    """Matrix factorisation with user/item biases, trained by mini-batch SGD.

    The rating of user :math:`u` for item :math:`i` is modelled as

    .. math:: \\hat r_{ui} = \\mu + b_u + b_i + p_u \\cdot q_i

    and the parameters minimise the L2-regularised squared error on the observed
    ratings only. Mini-batches keep the training vectorised: a pure-Python loop
    over the 5 million observed ratings would take hours.
    """

    def __init__(
        self,
        n_factors: int = 32,
        learning_rate: float = 0.01,
        reg: float = 0.05,
        n_epochs: int = 15,
        batch_size: int = 8192,
        rating_range: Tuple[float, float] = (1.0, 5.0),
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rating_range = rating_range
        self.seed = seed
        self.verbose = verbose

    def fit(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
        validation: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> "BiasedMF":
        """Fit the model on encoded (user index, item index, rating) triplets."""
        rng = np.random.default_rng(self.seed)
        ratings = np.asarray(ratings, dtype=np.float64)
        self.mu_ = float(ratings.mean())
        self.b_u_ = np.zeros(n_users)
        self.b_i_ = np.zeros(n_items)
        self.P_ = rng.normal(0.0, 0.05, size=(n_users, self.n_factors))
        self.Q_ = rng.normal(0.0, 0.05, size=(n_items, self.n_factors))
        self.history_: List[Dict[str, float]] = []

        n = len(ratings)
        lr, reg = self.learning_rate, self.reg
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                u, i, y = users[batch], items[batch], ratings[batch]
                p_u, q_i = self.P_[u], self.Q_[i]
                error = y - (
                    self.mu_
                    + self.b_u_[u]
                    + self.b_i_[i]
                    + np.einsum("ij,ij->i", p_u, q_i)
                )
                # `np.add.at` is the unbuffered version of `+=`: it accumulates
                # correctly when the same user or item appears twice in a batch.
                np.add.at(self.b_u_, u, lr * (error - reg * self.b_u_[u]))
                np.add.at(self.b_i_, i, lr * (error - reg * self.b_i_[i]))
                np.add.at(self.P_, u, lr * (error[:, None] * q_i - reg * p_u))
                np.add.at(self.Q_, i, lr * (error[:, None] * p_u - reg * q_i))

            entry = {
                "epoch": epoch,
                "train_rmse": rmse(ratings, self.predict(users, items)),
            }
            if validation is not None:
                entry["val_rmse"] = rmse(validation[2], self.predict(*validation[:2]))
            self.history_.append(entry)
            if self.verbose:
                print(
                    " | ".join(
                        f"{k}: {v:.4f}" if k != "epoch" else f"epoch {v:2.0f}"
                        for k, v in entry.items()
                    )
                )
        return self

    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Predict the ratings of the given (user index, item index) pairs."""
        raw = (
            self.mu_
            + self.b_u_[users]
            + self.b_i_[items]
            + np.einsum("ij,ij->i", self.P_[users], self.Q_[items])
        )
        return np.clip(raw, *self.rating_range)

    def score_all_items(self, user: int) -> np.ndarray:
        """Predicted rating of every item for a single encoded user index."""
        raw = self.mu_ + self.b_u_[user] + self.b_i_ + self.Q_ @ self.P_[user]
        return np.clip(raw, *self.rating_range)


class ItemKNNRecommender:
    """Item-based collaborative filtering on implicit (binary) feedback.

    Two books are similar when they were liked by the same people: the
    similarity is the cosine between the columns of the binary user-item
    matrix. Only the `k` nearest neighbours of each item are kept, which prunes
    the long tail of noisy similarities and keeps the model sparse.

    The score of an unseen item is the sum of its similarities to the items the
    user liked, i.e. one sparse product ``user_row @ similarity_matrix``.
    """

    def __init__(
        self,
        k: int = 100,
        like_threshold: float = 4.0,
        shrink: float = 0.0,
        block: int = 1000,
    ) -> None:
        self.k = k
        self.like_threshold = like_threshold
        self.shrink = shrink
        self.block = block

    def fit(
        self,
        users: np.ndarray,
        items: np.ndarray,
        ratings: np.ndarray,
        n_users: int,
        n_items: int,
    ) -> "ItemKNNRecommender":
        """Build the pruned item-item cosine similarity matrix."""
        liked = np.asarray(ratings) >= self.like_threshold
        users, items = np.asarray(users)[liked], np.asarray(items)[liked]
        self.interactions_ = csr_matrix(
            (np.ones(liked.sum()), (users, items)), shape=(n_users, n_items)
        )
        columns = normalize(self.interactions_.tocsc(), axis=0)
        popularity = np.asarray(self.interactions_.sum(axis=0)).ravel()

        rows, cols, values = [], [], []
        for start in range(0, n_items, self.block):
            stop = min(start + self.block, n_items)
            block = (columns[:, start:stop].T @ columns).toarray()
            if self.shrink:
                # Shrinkage penalises pairs supported by few common users.
                support = np.sqrt(np.outer(popularity[start:stop], popularity))
                block *= support / (support + self.shrink)
            np.fill_diagonal(block[:, start:stop], 0.0)
            n_neighbours = min(self.k, block.shape[1] - 1)
            candidates = np.argpartition(-block, n_neighbours, axis=1)[:, :n_neighbours]
            for row in range(stop - start):
                neighbours = candidates[row]
                similarity = block[row, neighbours]
                keep = similarity > 0
                rows.append(np.full(keep.sum(), start + row))
                cols.append(neighbours[keep])
                values.append(similarity[keep])

        self.similarity_ = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_items, n_items),
        )
        return self

    def score_all_items(self, user: int) -> np.ndarray:
        """Score every item for an encoded user index."""
        return np.asarray(
            (self.interactions_[user] @ self.similarity_).todense()
        ).ravel()

    def score_from_items(self, item_indices: Sequence[int]) -> np.ndarray:
        """Score the catalogue for a *new* user described by a list of items."""
        item_indices = list(item_indices)
        if not item_indices:
            return np.zeros(self.similarity_.shape[0])
        return np.asarray(self.similarity_[item_indices].sum(axis=0)).ravel()

    def similar_items(
        self, item_index: int, n: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the `n` most similar items to an encoded item index."""
        row = self.similarity_[item_index].toarray().ravel()
        top = np.argsort(-row)[:n]
        return top, row[top]


class ContentBasedRecommender:
    """TF-IDF content-based recommender over the book descriptions.

    Each book is a TF-IDF vector of its title/authors/tags; a user is the
    rating-weighted average of the books they liked. Scores are cosine
    similarities, so they live in ``[0, 1]`` and are *not* rating predictions.
    """

    def __init__(
        self,
        min_df: int = 3,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        like_threshold: float = 4.0,
    ) -> None:
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.like_threshold = like_threshold

    def fit(
        self, corpus: pd.Series, item_ids: Sequence[int]
    ) -> "ContentBasedRecommender":
        """Vectorise the corpus, restricted and ordered like `item_ids`."""
        self.item_ids_ = np.asarray(item_ids)
        documents = corpus.reindex(self.item_ids_).fillna("")
        self.vectorizer_ = TfidfVectorizer(
            min_df=self.min_df,
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.X_ = normalize(self.vectorizer_.fit_transform(documents.values))
        return self

    def similar_items(
        self, item_index: int, n: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the `n` closest items to an encoded item index (and scores)."""
        similarities = (self.X_ @ self.X_[item_index].T).toarray().ravel()
        similarities[item_index] = -np.inf
        top = np.argpartition(-similarities, n)[:n]
        top = top[np.argsort(-similarities[top])]
        return top, similarities[top]

    def build_user_profiles(
        self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray, n_users: int
    ) -> None:
        """Compute one profile vector per user out of the items they liked."""
        liked = ratings >= self.like_threshold
        weights = ratings[liked] - self.like_threshold + 1.0
        # One sparse product accumulates every liked book into the user rows.
        interaction = csr_matrix(
            (weights, (users[liked], items[liked])), shape=(n_users, self.X_.shape[0])
        )
        self.profiles_ = normalize(interaction @ self.X_)

    def score_all_items(self, user: int) -> np.ndarray:
        """Cosine similarity between the user profile and every item."""
        return np.asarray((self.profiles_[user] @ self.X_.T).todense()).ravel()

    def score_from_items(self, item_indices: Sequence[int]) -> np.ndarray:
        """Score the catalogue for a *new* user described by a list of items."""
        item_indices = list(item_indices)
        if not item_indices:
            return np.zeros(self.X_.shape[0])
        profile = normalize(np.asarray(self.X_[item_indices].mean(axis=0)))
        return np.asarray(profile @ self.X_.T).ravel()


class HybridRecommender:
    """Weighted hybrid of two scorers: ``alpha * first + (1 - alpha) * second``.

    Both score vectors are min-max normalised per user before being blended:
    a predicted rating (1-5), a similarity sum and a cosine similarity (0-1)
    are otherwise not comparable.
    """

    def __init__(self, collaborative, content, alpha: float = 0.7) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.collaborative = collaborative
        self.content = content
        self.alpha = alpha

    @staticmethod
    def _normalise(scores: np.ndarray) -> np.ndarray:
        low, high = float(scores.min()), float(scores.max())
        if high - low < 1e-12:
            return np.zeros_like(scores)
        return (scores - low) / (high - low)

    def score_all_items(self, user: int) -> np.ndarray:
        """Blended score of every item for an encoded user index."""
        collaborative = self._normalise(self.collaborative.score_all_items(user))
        content = self._normalise(self.content.score_all_items(user))
        return self.alpha * collaborative + (1.0 - self.alpha) * content

    def score_from_items(self, item_indices: Sequence[int]) -> np.ndarray:
        """Blended score for a *new* user described by a list of items."""
        collaborative = self._normalise(
            self.collaborative.score_from_items(item_indices)
        )
        content = self._normalise(self.content.score_from_items(item_indices))
        return self.alpha * collaborative + (1.0 - self.alpha) * content


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_recall_at_k(
    recommended: Sequence[int], relevant: Iterable[int], k: int
) -> Tuple[float, float]:
    """Precision@k and Recall@k of a ranked list against a relevant set."""
    relevant = set(relevant)
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for item in list(recommended)[:k] if item in relevant)
    recall = hits / len(relevant) if relevant else 0.0
    return hits / k, recall


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Binary-relevance NDCG@k of a ranked list."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    gains = np.array(
        [1.0 if item in relevant else 0.0 for item in list(recommended)[:k]]
    )
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = min(len(relevant), k)
    idcg = float(np.sum(1.0 / np.log2(np.arange(2, ideal + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    model,
    train_items_by_user: Dict[int, np.ndarray],
    relevant_by_user: Dict[int, np.ndarray],
    k: int = 10,
    users: Optional[Sequence[int]] = None,
    n_items: Optional[int] = None,
) -> Dict[str, float]:
    """Average Precision@k, Recall@k, NDCG@k and catalogue coverage.

    For every user the model scores the whole catalogue, the items already seen
    in the train set are removed, and the top-k list is compared with the items
    the user rated positively in the test set.

    Parameters
    ----------
    model : object
        Anything exposing ``score_all_items(user_index) -> np.ndarray``.
    train_items_by_user, relevant_by_user : dict
        Encoded item indices seen in training / rated positively in test.
    k : int
        Length of the recommendation list.
    users : sequence of int, optional
        Users to evaluate on (default: all users of `relevant_by_user`).
    n_items : int, optional
        Catalogue size, used for the coverage metric.

    Returns
    -------
    dict
        ``precision@k``, ``recall@k``, ``ndcg@k``, ``coverage`` and ``n_users``.
    """
    users = list(relevant_by_user) if users is None else list(users)
    precisions, recalls, ndcgs = [], [], []
    recommended_items: set = set()

    for user in users:
        relevant = relevant_by_user.get(user)
        if relevant is None or len(relevant) == 0:
            continue
        scores = np.array(model.score_all_items(user), dtype=float, copy=True)
        seen = train_items_by_user.get(user)
        if seen is not None and len(seen):
            scores[seen] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]

        precision, recall = precision_recall_at_k(top, relevant, k)
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg_at_k(top, relevant, k))
        recommended_items.update(top.tolist())

    coverage = len(recommended_items) / n_items if n_items else float("nan")
    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": coverage,
        "n_users": len(precisions),
    }


def evaluate_cold_start(
    model,
    profiles_by_user: Dict[int, Sequence[int]],
    relevant_by_user: Dict[int, np.ndarray],
    k: int = 10,
    n_items: Optional[int] = None,
) -> Dict[str, float]:
    """Same metrics as `evaluate_ranking`, for users described by a short profile.

    This is how a *new* user is served: nothing was learnt about them at
    training time, the only input is the handful of books they just told us
    they liked.

    Parameters
    ----------
    model : object
        Anything exposing ``score_from_items(item_indices) -> np.ndarray``.
    profiles_by_user : dict
        Encoded item indices describing each simulated new user.
    relevant_by_user : dict
        Encoded item indices the user actually liked (ground truth).
    """
    precisions, recalls, ndcgs = [], [], []
    recommended_items: set = set()

    for user, profile in profiles_by_user.items():
        relevant = relevant_by_user.get(user)
        if relevant is None or len(relevant) == 0:
            continue
        scores = np.array(model.score_from_items(profile), dtype=float, copy=True)
        scores[list(profile)] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]

        precision, recall = precision_recall_at_k(top, relevant, k)
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg_at_k(top, relevant, k))
        recommended_items.update(top.tolist())

    coverage = len(recommended_items) / n_items if n_items else float("nan")
    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": coverage,
        "n_users": len(precisions),
    }


def top_n_recommendations(
    model,
    user: int,
    index: IdIndex,
    books: pd.DataFrame,
    seen_items: Optional[Dict[int, np.ndarray]] = None,
    n: int = 10,
    columns: Sequence[str] = ("title", "authors", "average_rating"),
) -> pd.DataFrame:
    """Readable top-N recommendation list for an encoded user index.

    Parameters
    ----------
    model : object
        Anything exposing ``score_all_items(user_index)``.
    user : int
        Encoded user index.
    index : IdIndex
        The mapping between raw ids and encoded indices.
    books : pandas.DataFrame
        The books table, indexed by ``book_id``.
    seen_items : dict, optional
        Items already rated by the user, excluded from the list.
    n : int
        Length of the list.

    Returns
    -------
    pandas.DataFrame
        The `n` recommended books with their score.
    """
    scores = np.array(model.score_all_items(user), dtype=float, copy=True)
    if seen_items is not None and len(seen_items.get(user, [])):
        scores[seen_items[user]] = -np.inf
    top = np.argpartition(-scores, n)[:n]
    top = top[np.argsort(-scores[top])]

    recommendations = books.loc[index.items[top], list(columns)].copy()
    recommendations.insert(0, "rank", np.arange(1, len(top) + 1))
    recommendations["score"] = scores[top]
    return recommendations.reset_index()
