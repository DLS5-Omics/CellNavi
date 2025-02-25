# CellNavi

CellNavi is a deep learning framework designed to predict genes driving cellular transitions.

## Getting Started

### Installation

1. Clone the repo

```sh
git clone https://github.com/DLS5-Omics/CellNavi.git
```

2. Create conda environment

```sh
conda create -n cellnavi python=3.12
conda activate cellnavi
```

3. Install python dependencies

```sh
cd ./CellNavi
pip install -r requirements.txt
```

## Tutorials

Please refer to `tutorials/README.md` and `tutorials/Tutorial_perturbation.ipynb`. 

## Datasets provided for tutorial

The full descriptions of the datasets and the studies of origin can be found in the manuscript. Here we provide the links to the pretrained model and the example datasets.

### Pretrained model

- Pretrained model checkpoint and gene2token file: [link](https://www.dropbox.com/scl/fo/khjdwuvc9gczr97dl3o2i/AGEb-jDCWPqOMTzxIfFNGU8?rlkey=n8c7w54fqyty9sgrv47sdaphl&st=qj1drwjw&dl=0)


### Example datasets

- Example training and testing datasets and model path file under 1,000 training step, with adjacency matrix graph and shortest path graph provided: [link](https://www.dropbox.com/scl/fo/rq9klah7vqksn6e66dsae/AK3DJ2sxwL3MoWCOcQ9ZfFE?rlkey=1t4kz2vraif0ifu72c6gmo6xl&st=gpvwfw3j&dl=0)


## Citation

Wang, T., Pan, Y., Ju, F., Zheng, S., Liu, C., Min, Y., Liu, X., Xia, H., Liu, G., Liu, H., \& Deng, P. (2024). Directing cellular transitions on gene graph-enhanced cell state manifold. bioRxiv, 2024.10.27.620174; doi: [link](https://doi.org/10.1101/2024.10.27.620174)