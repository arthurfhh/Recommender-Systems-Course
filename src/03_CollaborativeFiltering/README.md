# Session 03 — Collaborative Filtering

## Lecture

* `03.CollaborativeFiltering.html` — the slides.
* `03_CollaborativeFiltering_Theory.ipynb` — the reading companion: notation, the cost
  function, the custom training loop.

## TP — Collaborative filtering from scratch on a MovieLens subset

`philippe.arthur-tp3.ipynb` (the former `03_CollaborativeFiltering_TP.ipynb`) is the TP, and it
is **completed and executed** in this repository: every `TODO` is implemented, the notebook is
committed with its outputs, and the results are discussed in the markdown cells that follow
them.

| Exercise | What was done |
|---|---|
| 1-2. Dataset and loaders | `load_precalc_params_small`, `load_ratings_small`, `load_movie_list`, plus a summary of the matrix (98 % empty, half of the movies with at most 2 ratings) |
| 3. Cost function, loops | `cofi_cost_func` — returns the expected 13.67 / 28.09 |
| 4. Cost function, vectorised | `cofi_cost_func_vectorized` — same values, checked against the loops on the full matrix; what forgetting the `R` mask costs; timing |
| 5. Rating movies | `normalize_ratings`, and 20 ratings entered by title (`build_my_ratings`) rather than by row index |
| 6. Training | The `GradientTape` loop of the statement, wrapped in `train_cofi` (same seed and hyper-parameters) with the cost curve |
| 7. Recommendations | `top_recommendations`, the fit on my own ratings, and the same list without the `min_ratings` filter |
| 8. Comparison table | `log_result`, writing `comparison_table.csv` in the format Session 04 uses |
| 9. Going further | All three topics — see below |

### Running it

```bash
pip install -r requirements.txt                # from the repository root
bash scripts/download_movielens_small.sh       # ~0.5 MB, into src/03_CollaborativeFiltering/data (git-ignored)
jupyter lab src/03_CollaborativeFiltering/philippe.arthur-tp3.ipynb
```

A full run takes about 7 minutes on a laptop CPU (18 trainings of 200 iterations); nothing
needs a GPU. The notebook was executed with Python 3.13 and TensorFlow 2.21.

### Note on the download

The statement downloads the archive with `megadl`, which was not installed on the machine the notebook was run on.
`scripts/download_movielens_small.sh` uses `megadl` when it is installed and otherwise
fetches the same MEGA link with `curl`, decrypting the archive with `openssl`
(AES-128-CTR), so the notebook runs wherever Git Bash, `curl` and `openssl` are available.

### What section 9 shows

| Topic | Result |
|---|---|
| **Regularisation** | 10 % of the ratings are held out. λ = 0 overfits completely (validation RMSE 3.90, predictions between −39 and +43); **λ = 1, the value of the statement, predicts unseen ratings no better than the movie means (0.970 against 0.971)**; λ = 10 is the best (**0.854**); λ = 30 underfits (0.867, the latent vectors collapse to 0). |
| **Cold start, new user** | The model ranks by raw mean rating, i.e. recommends movies with a single 5★. A damped mean fixes the list; replaying my own onboarding shows the model beating the movie means from about 5 ratings on. |
| **Cold start, new movie** | The plain model predicts about 0★ (RMSE ≈ 4). Borrowing the latent vector of the movies with the closest *title* brings it to 0.60-0.80 for franchise movies, and does nothing — or slightly worse than the catalogue mean — for the others. |
| **Toward a hybrid** | A comparison of what each model contributes, three ways to combine them, and a blend whose weight grows with the number of ratings of the movie. |

The takeaway: a recommendation list that looks sensible proves little. The model trained as
prescribed gives a convincing top 10, and held-out ratings show that it has learned nothing
that generalises beyond the movie means until the regularisation is made ten times stronger.

### One thing to personalise

The 20 ratings of section 5 (`MY_RATINGS_BY_TITLE`) are the profile the recommendations are
computed for. Change them and re-run the notebook to get your own list; the discussion cells
quote the numbers obtained with the committed ratings.
