"""Tests for `utils.book_recommender`.

They run on a small synthetic dataset, so they need neither the GoodBooks-10k
files nor a network connection.

Run them with `pytest src/tests/test_book_recommender.py` from the repository
root, or directly with `python src/tests/test_book_recommender.py`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import book_recommender as br  # noqa: E402


def make_ratings(n_users: int = 60, n_items: int = 25, seed: int = 0) -> pd.DataFrame:
    """Synthetic ratings with two latent groups of users and items."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(1, n_users + 1):
        group = user % 2
        for item in range(1, n_items + 1):
            if (item % 2) == group or rng.random() < 0.2:
                rating = 5 if (item % 2) == group else 2
                rows.append((user, item, rating))
    return pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def test_split_keeps_every_user_in_train():
    ratings = make_ratings()
    train, test = br.train_test_split_by_user(ratings, test_size=0.2, seed=1)

    assert len(train) + len(test) <= len(ratings)
    assert set(test.user_id) <= set(train.user_id)
    assert set(test.book_id) <= set(train.book_id)
    # No rating ends up on both sides of the split.
    merged = train.merge(test, on=["user_id", "book_id"], how="inner")
    assert merged.empty


def test_split_respects_min_train_ratings():
    ratings = pd.DataFrame(
        {"user_id": [1, 1, 2, 2, 2, 2], "book_id": [1, 2, 1, 2, 3, 4], "rating": 5}
    )
    train, _ = br.train_test_split_by_user(
        ratings, test_size=0.5, min_train_ratings=3, seed=0
    )
    assert (train.groupby("user_id").size() >= 2).all()


# --------------------------------------------------------------------------- #
# Index and grouping helpers
# --------------------------------------------------------------------------- #
def test_id_index_roundtrip():
    ratings = make_ratings(n_users=10, n_items=5)
    index = br.IdIndex.from_ratings(ratings)

    assert index.n_users == ratings.user_id.nunique()
    assert index.n_items == ratings.book_id.nunique()
    encoded = index.encode_items(ratings.book_id.head(5))
    assert (index.items[encoded] == ratings.book_id.head(5).to_numpy()).all()


def test_group_items_by_user():
    users = np.array([0, 0, 2, 1])
    items = np.array([3, 4, 5, 6])
    grouped = br.group_items_by_user(users, items, n_users=3)

    assert sorted(grouped[0].tolist()) == [3, 4]
    assert grouped[1].tolist() == [6]
    assert grouped[2].tolist() == [5]

    masked = br.group_items_by_user(
        users, items, n_users=3, mask=np.array([1, 0, 1, 0], bool)
    )
    assert masked[0].tolist() == [3]
    assert 1 not in masked


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_precision_recall_at_k():
    precision, recall = br.precision_recall_at_k([1, 2, 3, 4], relevant={2, 4, 9}, k=4)
    assert np.isclose(precision, 0.5)
    assert np.isclose(recall, 2 / 3)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert np.isclose(br.ndcg_at_k([1, 2, 3], relevant={1, 2, 3}, k=3), 1.0)
    assert br.ndcg_at_k([9, 8, 1], relevant={1, 2, 3}, k=3) < 1.0
    assert br.ndcg_at_k([9, 8, 7], relevant={1}, k=3) == 0.0


def test_ndcg_rewards_putting_hits_first():
    assert br.ndcg_at_k([1, 9, 9], {1}, k=3) > br.ndcg_at_k([9, 9, 1], {1}, k=3)


def test_rmse_and_mae():
    assert np.isclose(br.rmse([3.0, 5.0], [4.0, 4.0]), 1.0)
    assert np.isclose(br.mae([3.0, 5.0], [4.0, 4.0]), 1.0)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def test_popularity_damping_penalises_rare_items():
    ratings = pd.DataFrame(
        {
            # Book 1 and book 2 share the same 5.0 mean, but book 1 is backed by
            # five ratings and book 2 by a single one.
            "user_id": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "book_id": [1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3],
            "rating": [5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3],
        }
    )
    model = br.PopularityRecommender(damping=5.0).fit(ratings)

    assert np.isclose(model.global_mean_, 4.0)
    assert model.scores_[1] > model.scores_[2]  # damping shrinks the rare book
    assert model.top_n(n=1).index.tolist() == [1]
    assert model.ranker([1, 2, 3], by="count").scores.tolist() == [5.0, 1.0, 6.0]
    assert model.ranker([1, 2, 3], by="score").scores[0] > model.scores_[3]


def test_biased_mf_beats_the_global_mean():
    ratings = make_ratings()
    train, test = br.train_test_split_by_user(ratings, test_size=0.2, seed=3)
    index = br.IdIndex.from_ratings(train)

    u_tr, i_tr = index.encode_users(train.user_id), index.encode_items(train.book_id)
    u_te, i_te = index.encode_users(test.user_id), index.encode_items(test.book_id)
    y_tr, y_te = train.rating.to_numpy(float), test.rating.to_numpy(float)

    model = br.BiasedMF(
        n_factors=8,
        n_epochs=40,
        learning_rate=0.05,
        batch_size=256,
        seed=0,
        verbose=False,
    ).fit(u_tr, i_tr, y_tr, index.n_users, index.n_items)

    baseline = br.rmse(y_te, np.full_like(y_te, y_tr.mean()))
    assert br.rmse(y_te, model.predict(u_te, i_te)) < baseline
    assert len(model.history_) == 40
    predictions = model.predict(u_te, i_te)
    assert predictions.min() >= 1.0 and predictions.max() <= 5.0


def test_item_knn_recovers_the_latent_groups():
    ratings = make_ratings()
    index = br.IdIndex.from_ratings(ratings)
    users = index.encode_users(ratings.user_id)
    items = index.encode_items(ratings.book_id)

    model = br.ItemKNNRecommender(k=5).fit(
        users, items, ratings.rating.to_numpy(float), index.n_users, index.n_items
    )
    # Books of the same parity were liked by the same users, so they must be
    # each other's nearest neighbours.
    neighbours, _ = model.similar_items(index.item_pos[1], n=3)
    assert all(index.items[n] % 2 == 1 for n in neighbours)

    scores = model.score_all_items(index.user_pos[1])
    assert scores.shape == (index.n_items,)
    assert np.isfinite(scores).all()


def test_content_based_scores_are_cosine_similarities():
    books = pd.DataFrame(
        {
            "book_id": [1, 2, 3],
            "goodreads_book_id": [10, 20, 30],
            "title": ["Dune", "Dune Messiah", "French Cooking"],
            "original_title": ["Dune", "Dune Messiah", "French Cooking"],
            "authors": ["Frank Herbert", "Frank Herbert", "Julia Child"],
        }
    )
    book_tags = pd.DataFrame(
        {"goodreads_book_id": [10, 20, 30], "tag_id": [1, 1, 2], "count": [100, 90, 80]}
    )
    tags = pd.DataFrame({"tag_id": [1, 2], "tag_name": ["science-fiction", "cookbook"]})

    corpus = br.build_content_corpus(books, book_tags, tags)
    assert "frank_herbert" in corpus.loc[1]
    assert "science-fiction" in corpus.loc[1]

    model = br.ContentBasedRecommender(min_df=1).fit(corpus, [1, 2, 3])
    neighbours, similarities = model.similar_items(0, n=2)
    assert neighbours[0] == 1  # "Dune Messiah" is the closest book to "Dune"
    assert 0.0 <= similarities[0] <= 1.0 + 1e-9
    assert similarities[0] > similarities[1]

    scores = model.score_from_items([0])
    assert scores[1] > scores[2]


def test_hybrid_interpolates_between_its_two_models():
    class Fake:
        def __init__(self, scores):
            self.scores = np.asarray(scores, float)

        def score_all_items(self, user):
            return self.scores

        def score_from_items(self, items):
            return self.scores

    collaborative, content = Fake([0.0, 1.0, 2.0]), Fake([2.0, 1.0, 0.0])

    assert np.allclose(
        br.HybridRecommender(collaborative, content, alpha=1.0).score_all_items(0),
        [0.0, 0.5, 1.0],
    )
    assert np.allclose(
        br.HybridRecommender(collaborative, content, alpha=0.5).score_all_items(0),
        [0.5, 0.5, 0.5],
    )
    assert np.allclose(
        br.HybridRecommender(collaborative, content, alpha=0.0).score_from_items([0]),
        [1.0, 0.5, 0.0],
    )


# --------------------------------------------------------------------------- #
# Evaluation loops
# --------------------------------------------------------------------------- #
def test_evaluate_ranking_masks_already_seen_items():
    class Fake:
        def score_all_items(self, user):
            return np.array([10.0, 9.0, 8.0, 7.0])

    seen = {0: np.array([0, 1])}
    relevant = {0: np.array([2])}
    metrics = br.evaluate_ranking(Fake(), seen, relevant, k=2, n_items=4)

    # Items 0 and 1 are masked, so the list is [2, 3] and item 2 is a hit.
    assert np.isclose(metrics["precision@2"], 0.5)
    assert np.isclose(metrics["recall@2"], 1.0)
    assert np.isclose(metrics["ndcg@2"], 1.0)
    assert np.isclose(metrics["coverage"], 0.5)
    assert metrics["n_users"] == 1


def test_evaluate_cold_start_excludes_the_profile_items():
    class Fake:
        def score_from_items(self, items):
            return np.array([10.0, 9.0, 8.0, 7.0])

    metrics = br.evaluate_cold_start(
        Fake(), {0: [0]}, {0: np.array([1])}, k=2, n_items=4
    )
    assert np.isclose(metrics["precision@2"], 0.5)
    assert metrics["n_users"] == 1


def test_top_n_recommendations_is_readable():
    books = pd.DataFrame(
        {
            "title": ["A", "B", "C"],
            "authors": ["x", "y", "z"],
            "average_rating": [4.0, 3.0, 5.0],
        },
        index=pd.Index([1, 2, 3], name="book_id"),
    )
    index = br.IdIndex(users=np.array([7]), items=np.array([1, 2, 3]))

    class Fake:
        def score_all_items(self, user):
            return np.array([1.0, 3.0, 2.0])

    table = br.top_n_recommendations(
        Fake(), 0, index, books, seen_items={0: np.array([1])}, n=2
    )
    assert table["rank"].tolist() == [1, 2]
    assert table["title"].tolist() == ["C", "A"]  # book 2 was already read
    assert table["score"].is_monotonic_decreasing


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"\033[92mPASSED\033[0m {name}")
            except AssertionError as error:
                failures += 1
                print(f"\033[91mFAILED\033[0m {name}: {error}")
    raise SystemExit(failures)
