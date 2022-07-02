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

The 25M version is the long-term target and matches what the original paper
benchmarks on for scale; 100K is the tractable default.

## Status

Work in progress. README will fill in as the model lands.

## License

MIT, see `LICENSE`.
