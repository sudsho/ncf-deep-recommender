# ncf-deep-recommender

Neural Collaborative Filtering (NCF) implementation on the MovieLens dataset.
Trains GMF, MLP and the fused NeuMF variant from He et al. 2017,
evaluates with leave-one-out HR@K and NDCG@K, and exposes a top-N recommendation
endpoint via FastAPI.

## Why

I wanted a hands-on, end-to-end implementation of the NCF paper to compare
the three variants (matrix-factorisation style GMF, deep MLP, and the
NeuMF fusion) on the same data and pipeline. Plus a small inference service
on top so the model is actually usable instead of stuck inside a notebook.

## Dataset

By default this uses **MovieLens-100K** (small enough to keep the repo light and
let CI run training-on-a-tiny-slice tests in seconds). The code will also work
on **MovieLens-25M** if you point `data.path` in the config to the bigger file.
See `data/README.md` for the download script.

Switch dataset by editing `data.path` in `configs/default.yaml`:

```yaml
data:
  path: data/ml-100k/u.data   # default, small
  # path: data/ml-25m/ratings.csv   # full 25M, slower
```

## Architectures

Three model variants from He et al. 2017, all in `src/model.py`:

- **GMF**: element-wise product of user and item embeddings, then a linear head.
- **MLP**: concat user/item embeddings, push through a few dense layers.
- **NeuMF**: GMF and MLP towers in parallel, concatenate before the head.

Training is single-pass binary cross-entropy on positives sampled from the
ratings file plus uniform negatives drawn from the user's unseen items.
Negatives are resampled every epoch.

## Eval

Leave-one-out: hold out the most recent interaction per user, score it
against 99 sampled negatives, compute HR@10 and NDCG@10. Same protocol as
the paper.

Sample run on ml-100k (5 epochs, NeuMF, embed_dim_gmf=16, mlp_layers [64,32,16,8]):

| Model  | HR@10 | NDCG@10 |
|--------|-------|---------|
| GMF    | 0.61  | 0.34    |
| MLP    | 0.65  | 0.37    |
| NeuMF  | 0.69  | 0.41    |

(Numbers will move around with seeds and epochs; these are rough.)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh
```

## Train

```bash
python -m src.train --config configs/default.yaml
# or the slightly bigger NeuMF config
python -m src.train --config configs/neumf.yaml
```

## Recommend

After training, hit the FastAPI endpoint (added in `src/api/main.py`):

```bash
uvicorn src.api.main:app --reload
# 5 dense-index items, masking what the user already saw
curl 'http://localhost:8000/recommend?user_id=12&n=5'

# original MovieLens ids, useful when joining against a movies metadata file
curl 'http://localhost:8000/recommend?user_id=12&n=5&original_ids=true'
```

The service caches the model on first request, so cold start is ~200 ms and
subsequent requests are millisecond-scale on CPU for ml-100k.

## Run with Docker

Train locally first so `artifacts/` is populated, then:

```bash
docker compose up --build
# api on :8000, mlflow ui on :5000
curl 'http://localhost:8000/health'
curl 'http://localhost:8000/info'
curl 'http://localhost:8000/recommend?user_id=12&n=5'
```

The compose file mounts `./artifacts/` read-only into the container so the
service picks up the latest checkpoint without rebuilding.

## Project layout

```
src/
  data.py        # ratings loader, leave-one-out split, neg sampling, Dataset
  model.py       # GMF, MLP, NeuMF + factory
  train.py       # config-driven training loop with mlflow logging
  evaluate.py    # HR@K + NDCG@K (loo with sampled negatives)
  predict.py     # top-N for a user from a saved checkpoint
  api/
    main.py      # FastAPI app
    schemas.py   # pydantic v1 schemas
configs/
  default.yaml   # ml-100k baseline
  neumf.yaml     # bigger NeuMF run
  tiny.yaml      # used by smoke / CI
tests/           # pytest suite
notebooks/eda.ipynb
scripts/         # download_data.sh, make_tiny_data.py
```

## License

MIT, see `LICENSE`.
