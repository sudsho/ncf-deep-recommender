#!/usr/bin/env bash
# Download MovieLens 100K into ./data/ml-100k.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -d "ml-100k" ]; then
    echo "ml-100k already present, skipping."
    exit 0
fi

URL="https://files.grouplens.org/datasets/movielens/ml-100k.zip"
echo "fetching $URL"
curl -sSL -o ml-100k.zip "$URL"
unzip -q ml-100k.zip
rm ml-100k.zip
echo "done -> $DATA_DIR/ml-100k"
