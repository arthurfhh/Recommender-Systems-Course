# Session 04 — Evaluation Metrics

## Lecture

* `04.EvaluationMetrics.html` — the slides.

## TP — An offline evaluation pipeline

`philippe.arthur-tp4.ipynb` (the former `04_EvaluationMetrics_TP.ipynb`) is the TP, and it is
**completed and executed** in this repository: every `TODO` is implemented, the notebook is
committed with its outputs, and the results are discussed in the markdown cells that follow
them.

| Exercise | What was done |
|---|---|
| 1. Temporal split | `temporal_user_split` — stable sort, `max(1, floor(0.2 n))` held out per user; gives the 80 896 / 19 940 rows the statement announces, and the no-leakage assertion passes |
| 2. Ranking metrics | `precision_at_k`, `recall_at_k`, `ndcg_at_k` — the five expected values of the self-test are matched exactly |
| 3. Error metrics | `rmse`, `mae` — asserts pass; global mean 1.0571 / 0.8275, item mean 1.0061 / 0.7736 |
| 4. Beyond accuracy | `catalog_coverage`, `intra_list_diversity`, `novelty` — the five expected values are matched exactly |
| 5. Evaluation loop | `evaluate_recommender` — 590 evaluable users, five per-user metrics, deterministic user order |
| 8. Comparison | The seven recommenders, **plus** a paired per-user comparison of Precision@10 against the best model |
| 9. Comparison table | `comparison_table.csv` written next to the notebook (seven S04 rows) |
| 10. Going further | Topic 1 — the MMR trade-off, with paired significance tests (see below) |

### Running it

```bash
pip install -r requirements.txt                 # from the repository root
jupyter lab src/04_EvaluationMetrics/philippe.arthur-tp4.ipynb
```

The notebook is **self-contained**: it needs no other file of the repository and downloads its
dataset (~2.8 MB) into `./data` next to itself. A full run takes about 40 seconds on a laptop
CPU; nothing needs a GPU. It was checked by copying the notebook alone into an empty folder and
executing it there from top to bottom.

### Note on the download

`files.grouplens.org` answers `403 Forbidden` from some networks — including the machine this
notebook was executed on, where the egress proxy denies the tunnel. `download_movielens` keeps
the URL of the statement as its first choice and falls back to a GitHub mirror of the same
release, then asserts the dataset size (100 836 ratings, 610 users, 9 742 movies) so a wrong
mirror cannot pass unnoticed.

### What the comparison shows

| Model | Precision@10 | Coverage | Diversity | Novelty |
|---|---|---|---|---|
| **Most popular** | **0.0563** | 0.012 | 0.740 | 1.52 |
| Best-rated (min 50) [S01] | 0.0376 | 0.008 | 0.718 | 2.63 |
| Content + prior + MMR 0.5 | 0.0356 | 0.073 | 0.398 | 3.34 |
| Content + popularity prior | 0.0342 | 0.085 | 0.225 | 3.57 |
| Content (genres) | 0.0137 | 0.117 | 0.121 | 5.69 |
| Random | 0.0019 | 0.513 | 0.805 | 7.47 |
| Best-rated (min 1) | 0.0003 | 0.002 | 0.781 | 9.15 |

Two results worth keeping:

* **No personalised model beats the non-personalised baseline.** `Most popular` leads on
  precision, recall and NDCG, and the paired tests put every other model at |t| ≥ 4.8 below it.
  With only the 19 MovieLens genres as features, content-based filtering has less information
  than "what everybody watched". What the beyond-accuracy columns add is the reason to ship a
  personalised model anyway: the popularity shelf serves 1.2 % of the catalogue.
* **`Best-rated (min 1)` scores below random** (0.0003 against 0.0019) with the highest novelty
  of the table: ranking by mean rating without a support threshold fills the list with items
  rated 5.0 by one single user. The same formula with a 50-rating threshold is a hundred times
  more precise — the eligibility rule matters more than the ranking criterion.

### What section 10 shows

The statement asks for at least one of four topics; the one treated is the **MMR trade-off**.
Five values of λ, each evaluated on the same 590 users, compared with **paired** standard
errors rather than independent ones:

| λ | Precision@10 | Diversity |
|---|---|---|
| 0.9 | 0.0349 | 0.281 |
| **0.7** | **0.0383** | 0.367 |
| 0.5 | 0.0356 | 0.398 |
| 0.3 | 0.0349 | 0.407 |
| 0.1 | 0.0337 | 0.409 |

Diversity moves monotonically (+45 %, t = −33 between the extremes) while precision does not
move at all in a measurable way (Δ = +0.0012 ± 0.0019, t = 0.62): on these numbers the extra
diversity is free. But λ = 0.7 is the best precision of the five and beats λ = 0.1
significantly (Δ = +0.0046 ± 0.0016, t = 2.88), because MMR breaks the ties between the
hundreds of items that share an identical genre vector. The notebook states the two caveats
that go with that claim: the selection bias of picking the best of five values, and the
U-shaped novelty/coverage curves.
