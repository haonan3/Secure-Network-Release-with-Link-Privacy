#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python graph_classification_exp/preprocess_nx_data.py --dataset relabeled_dblp2 --model orig

CUDA_VISIBLE_DEVICES=1 python graph_classification_exp/main.py --hidden_dim 256 --epochs 300 --lr 0.0005 --dataset relabeled_dblp2 &