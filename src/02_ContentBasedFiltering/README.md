# Session 02 — Content-Based Filtering

## Lecture

* `ContentBasedFiltering.ipynb` / `ContentBasedFiltering.html` — the lecture notebook and its
  slide export.
* `knn_recommender.py` — the KNN + TF-IDF demo used during the lecture.

## TP — Book recommender on GoodBooks-10k

| File | What it is |
|---|---|
| `BookRecommender_exercise.ipynb` | The assignment (statement only) |
| `philippe.arthur-tp2.ipynb` | Worked solution (ex-`BookRecommender_solution.ipynb`), executed, with all the outputs |
| `../utils/book_recommender.py` | The reusable code: loading, split, models, metrics |
| `../tests/test_book_recommender.py` | Unit tests for the above (synthetic data, no download) |
| `../../scripts/download_goodbooks.sh` | Downloads the dataset into `data/` (git-ignored) |

### Running it

```bash
pip install -r requirements.txt          # from the repository root
bash scripts/download_goodbooks.sh       # ~32 MB zipped, ~100 MB extracted
jupyter lab src/02_ContentBasedFiltering/philippe.arthur-tp2.ipynb
```

A full run takes about 6 minutes on a laptop CPU; nothing needs a GPU.
The tests run without the dataset:

```bash
python src/tests/test_book_recommender.py     # or: pytest src/tests/test_book_recommender.py
```

### What the solution covers

1. **Data analysis** — rating distribution, long tail, sparsity (98.9 % of the matrix is
   missing), missing-value decisions.
2. **Feature engineering** — id encoding, damped popularity, TF-IDF documents built from
   title + authors + tags.
3. **Evaluation** — per-reader 80/20 split, RMSE/MAE *and* Precision/Recall/NDCG@10 plus
   catalogue coverage.
4. **Models** — popularity baselines, biased matrix factorisation (mini-batch SGD),
   item-based collaborative filtering on implicit feedback, content-based TF-IDF, and a
   weighted hybrid.
5. **Ablations** — dataset size, content features, neighbourhood size, blend weight.
6. **Cold start** — new reader and new book.
7. **Recommendations** — top-N lists per reader and "because you liked…" item-to-item lists.

### Headline results (test set)

| Task | Best model | Score | Baseline |
|---|---|---|---|
| Rating prediction | Biased MF, 32 factors | RMSE **0.847** / MAE 0.662 | 0.991 / 0.774 (global mean) |
| Top-10 recommendation | Hybrid (0.85 item-kNN + 0.15 content) | NDCG@10 **0.363**, Precision@10 **0.313** | 0.090 / 0.080 (MostPopular) |
| Ranking books nobody rated yet | Content-based TF-IDF | Recall@10 **0.290** | 0.026 (item-kNN) |

The main takeaway of the TP: **the model with the best RMSE (matrix factorisation) produces
the worst top-10 lists** — thirty times less precise than recommending the most-rated books to
everybody. Optimising the squared error on observed ratings is not the same problem as ranking
a catalogue.
