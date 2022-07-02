# data

This folder is where the MovieLens data lives at runtime. The actual files
are gitignored so the repo stays small.

## ml-100k (default)

```bash
mkdir -p data
cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

Expected layout: `data/ml-100k/u.data` (tab separated, columns:
user_id, movie_id, rating, timestamp).

## ml-25m (optional, larger experiments)

```bash
cd data
curl -O https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
```

Then set `data.path: data/ml-25m/ratings.csv` in the config.
