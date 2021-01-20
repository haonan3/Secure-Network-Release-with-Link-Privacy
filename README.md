# Secure Network Release with Link Privacy

This repository is the PyTorch implementation of Secure Network Release with Link Privacy.

[arXiv](https://arxiv.org/abs/2005.00455)

If you make use of the code/experiment, please cite our paper (Bibtex below).

```
@misc{yang2020secure,
    title={Secure Network Release with Link Privacy},
    author={Carl Yang and Haonan Wang and Lichao Sun and Bo Li},
    year={2020},
    eprint={2005.00455},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
}
```

Contact: Haonan Wang (haonan3@illinois.edu), Carl Yang (yangji9181@gmail.com)


## Installation
Install PyTorch following the instuctions on the [official website] (https://pytorch.org/). The code has been tested over PyTorch 1.1.0 versions.

Then install the other dependencies.
```
conda env create -f environment.yml

conda activate dpggan

pip install -r requirements.txt
```

## Test run
Unzip the dataset file
```
unzip data.zip
```

and run

```
sh run.sh
```

Default parameters are not the best performing-hyper-parameters. Hyper-parameters need to be specified through the commandline arguments.

For graph classification experiment and link prediction experiment, please refer `run_graph_classification_exp.sh` and `run_link_classification_exp.sh`.