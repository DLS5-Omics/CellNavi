# CellNavi

CellNavi is a deep learning framework designed to predict genes driving cellular transitions. It is trained on large-scale, high-dimensional single-cell transcriptomic data, along with prior causal gene graphs that reveal the underlying structure of cell states. By projecting cellular data onto this biologically meaningful space with reduced dimensionality and enhanced biological relevance, CellNavi provides a universal framework that generalizes across diverse cellular contexts, with diverse applications include:

- identifying CRISPR target
- predicting mastergit regulator
- discovering disease driver
- elucidating drug MoA

<!-- ![overview](cellnavi/overview.png) -->
<p align="center">
  <img src="cellnavi/overview.png" width="80%">
</p>



## Installation

1. Create a new conda environment (Highly recommend!)

```sh
conda create -n cellnavi python=3.12
conda activate cellnavi
```

2. Clone the repo

```sh
git clone https://github.com/DLS5-Omics/CellNavi.git
```


3. Install the python package 'cellnavi'

```sh
cd ./CellNavi
pip install .
```

## Tutorials

Please refer to `tutorials/README.md` and `tutorials/Tutorial_perturbation.ipynb`. 


#### Pretrained model weights

- Pretrained model checkpoint and gene2token file: [link](https://www.dropbox.com/scl/fo/khjdwuvc9gczr97dl3o2i/AGEb-jDCWPqOMTzxIfFNGU8?rlkey=n8c7w54fqyty9sgrv47sdaphl&st=qj1drwjw&dl=0)


#### Example datasets

- Example training and testing datasets and model path file under 1,000 training step, with adjacency matrix graph and shortest path graph provided: [link](https://www.dropbox.com/scl/fo/rq9klah7vqksn6e66dsae/AK3DJ2sxwL3MoWCOcQ9ZfFE?rlkey=1t4kz2vraif0ifu72c6gmo6xl&st=gpvwfw3j&dl=0)




## Citation   
```
@article{wang2024directing,
  title={Directing cellular transitions on gene graph-enhanced cell state manifold},
  author={Wang, Tianze and Pan, Yan and Ju, Fusong and Zheng, Shuxin and Liu, Chang and Min, Yaosen and Liu, Xinwei and Xia, Huanhuan and Liu, Guoqing and Liu, Haiguang and others},
  journal={bioRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```