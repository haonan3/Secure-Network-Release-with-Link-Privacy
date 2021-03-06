#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python src/main.py --model_name GGAN --dataset new_dblp2 &

python src/main.py --model_name GGAN --dataset new_IMDB_MULTI  &

python src/main.py --model_name GVAE --dataset new_dblp2 &

python src/main.py --model_name GVAE --dataset new_IMDB_MULTI &