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
| 9. Going further | The regularisation study, on held-out ratings — see below |

### Running it

```bash
pip install -r requirements.txt                # from the repository root
jupyter lab src/03_CollaborativeFiltering/philippe.arthur-tp3.ipynb
```

The notebook is **self-contained**: it needs no other file of the repository and downloads
its dataset (~0.5 MB) into `./data` next to itself, as the statement does. A full run takes
about 2 minutes on a laptop CPU (5 trainings of 200 iterations); nothing needs a GPU.

It was checked by copying the notebook alone into an empty folder and executing it there from
top to bottom, twice: with Python 3.13 / TensorFlow 2.21, and with Python 3.9 / TensorFlow
2.14.1 and the pinned versions of `requirements.txt`. Both runs complete without error and
give the same numbers, which are the ones quoted in the discussion cells.

### Note on the download

The download cell is the one of the statement with a fallback added: when `megadl` is
installed it runs the original command; otherwise it fetches the same MEGA link with `curl`
and decrypts the archive with `openssl` (AES-128-CTR). The archive is then extracted by a
Python cell (`zipfile`) rather than with `unzip`. If `./data` already holds the csv files,
both cells do nothing.

### What section 9 shows

The statement asks for at least one of three topics; the one treated is the
**regularisation**. 10 % of the ratings are held out and the model is retrained with four
values of λ:

| λ | Validation RMSE | Reading |
|---|---|---|
| 0 | 3.90 | Complete overfitting: predictions between −39 and +43 |
| 1 (the statement's value) | 0.970 | **No better than predicting each movie's mean (0.971)** |
| 10 | **0.854** | Best of the four |
| 30 | 0.867 | Underfitting: the latent vectors collapse to 0 |

The takeaway: a recommendation list that looks sensible proves little. The model trained as
prescribed gives a convincing top 10, and held-out ratings show that it has learned nothing
that generalises beyond the movie means until the regularisation is made ten times stronger.

### One thing to personalise

The 20 ratings of section 5 (`MY_RATINGS_BY_TITLE`) are the profile the recommendations are
computed for. Change them and re-run the notebook to get your own list; the discussion cells
quote the numbers obtained with the committed ratings.
