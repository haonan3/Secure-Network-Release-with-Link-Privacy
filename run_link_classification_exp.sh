#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CUDA_VISIBLE_DEVICES=1 python link_classification_exp/main.py --graph_name new_IMDB_MULTI --graph_type 'GGAN_{}' --epochs 40 &

CUDA_VISIBLE_DEVICES=2 python link_classification_exp/main.py --graph_name new_dblp2 --graph_type 'GGAN_{}' --epochs 40 &