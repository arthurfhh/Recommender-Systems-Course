# Session 05 — Hybrid Recommenders

## Lecture

* `05.HybridNeuralEncoding.html` — the slides.

## TP — A neural hybrid: the two-tower network

`philippe.arthur-tp5.ipynb` (the former `05_HybridNeuralEncoding_TP.ipynb`) is the TP, and it
is **completed and executed** in this repository: every `TODO` is implemented, the notebook is
committed with its outputs, and the results are discussed in the markdown cells that follow
them.

| Exercise | What was done |
|---|---|
| 1. The features | `build_item_features` (genres, year, shrunk mean, log count) and `build_user_features` (per-genre mean rating, log count, mean) — the toy self-test matches to the fourth decimal, shapes `(8246, 22)` and `(610, 21)` as announced |
| 2. The network | `build_two_tower` — two MLP towers, unit-norm embeddings, dot product, optional sigmoid calibration; the five checks of the statement pass, weight sharing included |
| 3. The training pairs | `sample_training_pairs` — one positive per rating ≥ 4, `n_neg` negatives drawn among the items the user never rated, with replacement; the toy test matches |
| 4. The item neighbours | `similar_items` — cosine in embedding space, query item excluded; toy test matches |
| 8. Hybrid | Added: **paired** per-user tests of the hybrid against each component |
| 9. Comparison table | `comparison_table.csv` written next to the notebook (four S05 rows) |
| 6. Written questions | Answered in the quoted markdown cells, including the one the statement asks in section 6: what the Session 03 latent-factor model would have done with a film nobody has rated |
| 10. Going further | Topic 1 — how noisy is training, over five seeds (see below) |

### Running it

```bash
pip install -r requirements.txt     # from the repository root; it already pins tensorflow
jupyter lab src/05_HybridRecommenders/philippe.arthur-tp5.ipynb
```

The notebook is **self-contained**: it needs no other file of the repository and downloads its
dataset (~2.8 MB) into `./data` next to itself. A full run takes about **8 minutes** on a
laptop CPU (nine networks trained in total, five of them for the seed study); nothing needs a
GPU.

Checked by copying the notebook alone into an empty folder and executing it there from top to
bottom, with Python 3.11, TensorFlow 2.21 and Keras 3.15: no error, numbers as committed. The
only Keras-3-specific line is the `KERAS_BACKEND` one of the statement itself; everything the
exercises use (`UnitNormalization`, `Dot`, `set_random_seed`) exists since Keras 2.12.

### Note on the download

`files.grouplens.org` answers `403 Forbidden` from some networks — including the machine this
notebook was executed on, where the egress proxy denies the tunnel. As in the Session 04 TP,
`download_movielens` keeps the URL of the statement as its first choice and falls back to a
GitHub mirror of the same release; the cell prints the dataset size, so a wrong mirror cannot
pass unnoticed.

### What the results show

| Model | RMSE | Precision@10 | NDCG@10 |
|---|---|---|---|
| Popularity | — | 0.0563 | 0.0743 |
| Two-tower, **rating** objective | **0.892** | 0.0100 | 0.0115 |
| Two-tower, **retrieval** objective | — | 0.0576 | 0.0718 |
| Two-tower + popularity (weights 0.8 / 0.2 chosen on validation) | — | **0.0705** | **0.0920** |

* **The loss defines the model, not the architecture.** The same network trained to predict
  ratings reaches the best RMSE of the course (0.892 against 1.006 for the item mean) and ranks
  six times *worse* than popularity; trained on positives and sampled negatives, it ranks six
  times better. Session 04 showed that the metric has to match the task — here it is the
  training objective.
* **The weighted hybrid is the only system that beats popularity by more than the noise**
  (+0.0129 over the network, t = 4.03; +0.0142 over popularity, t = 4.06), while the network
  alone does not (+0.0014, t = 0.29). Its weight was chosen on a validation split of the
  training set, not on the test set.
* **Half of the network's ranking power is popularity.** Zeroing two of the 22 item features
  (shrunk mean and log count) costs 42 % of its precision and leaves it exactly where the
  Session 04 content model stood.
* **The item tower learns taste, not genres.** *Pulp Fiction*'s neighbours are *Fargo*,
  *The Shawshank Redemption* and *Schindler's List* — no shared genre, one shared audience.
  And because a tower reads features, it embeds a 2024 film that nobody has rated, which the
  Session 03 latent-factor model could not do at all.

### What section 10 shows

The statement asks for at least one of five topics; the one treated is the **training noise**.
The retrieval network is trained five times, changing only the seed:

| seed | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Precision@10 | 0.0576 | 0.0593 | 0.0659 | 0.0622 | 0.0595 |

The spread between seeds (sd 0.0032) is smaller than the standard error over users (0.0045)
but of the same order — and among the ten pairs of seeds, that is ten comparisons of the model
with *itself*, **one comes out "significant"** at |t| > 1.96. The network used in sections 5 to
8 is seed 1, the worst of the five, which is why the notebook restates the hybrid's advantage
as a range rather than a point. The rule drawn from it: compare architectures over several
seeds, compare fixed models with paired per-user tests, and never mix the two.
