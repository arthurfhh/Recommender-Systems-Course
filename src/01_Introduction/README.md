# Session 01 — Introduction

## Lecture

* `Introduction.ipynb` / `01.RecSysIntro.html` — what a recommender system is, the
  families of algorithms, the vocabulary used in the rest of the course.

## TP — Non-personalised recommendations on MovieLens

`MovieLensDataAnalysis.ipynb` is the TP, and it is **completed and executed** in
this repository: every `📝 TASK` / `Insert your code here` cell is filled in, and
the "❓ Think About" questions are answered in the `>` quoted cells.

| Exercise | Where | What was done |
|---|---|---|
| Choose and download a dataset | bash cell | MovieLens 10M for the "too big" part, MovieLens 100K for the rest |
| Reduce the dataset | `TASK` cell | ≥ 50 ratings per movie, ≥ 20 ratings per user, with the before/after comparison |
| Effect of the reduction | `TASK` cell | Sizes, matrix density, rating distribution before/after |
| Build the user-movie matrix | `Insert your code` | `pivot_table(index="UserId", columns="MovieId", values="Rating")` |
| Long tail on a bigger dataset | inserted cell | 100K vs 10M, log-log plot and summary table |
| `get_recent_liked_movie` | `Complete the function` | Passes `tests/non_pers_tests.py` |
| `get_genre` | `Complete the function` | Passes `tests/non_pers_tests.py` |
| `get_recommendations` | inserted cell | Combines the two, handles the unknown user |
| Godfather / exclude seen movies | inserted cells | `recommend_genre_n(..., exclude_movies=...)` + the check on the saga |
| Serendipity-based recommender | `Insert your code` | Popular / hidden-gem blend, restricted to the user's genres |

### Running it

```bash
pip install -r requirements.txt     # from the repository root
bash scripts/download_movielens.sh  # ml-100k + ml-10m, into src/data (git-ignored)
jupyter lab src/01_Introduction/MovieLensDataAnalysis.ipynb
```

A full run takes a bit more than a minute.

### Three notes on the environment

1. **`files.grouplens.org` may be unreachable** (it is blocked by the network
   policy of the machine this notebook was executed on). `scripts/download_movielens.sh`
   tries GroupLens first and falls back to a GitHub mirror of the same archives;
   `utils/data_utils.py` now reads the local copy when there is one and keeps the
   GroupLens URLs as the fallback, so the notebook runs offline.
2. **MovieLens 10M stands in for the multi-million-rating dataset** the notebook
   originally pointed at (`ml-latest`), which the mirror does not carry. The cell
   that loads it reads `../data/movielens_complete/ml-10m/{ratings,movies}.csv`.
3. **`utils/non_pers_rec.py` had an off-by-two on the genre columns**
   (`df_items.columns[7:]` starts at *Animation*, dropping *Action* and
   *Adventure*). It is now `[5:]`, which matches the genre list used in the
   notebook — without this fix the `get_genre` unit test can never pass.
