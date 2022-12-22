# cblearn-benchmark

This repository contains a small empirical comparison of algorithm implementations in [cblearn](https://github.com/dekuenstle/cblearn) 
with each other and with implementations in different libraries. 

At the moment, only ordinal embedding algorithms are evaluated. 


## Preparation

Setup a conda environment:
```
conda create -n cblearn python==3.10
conda activate cblearn

conda install h5py seaborn tqdm pandas
conda install -c conda-forge adjusttext
pip install git+https://github.com/dekuenstle/cblearn.git#egg=cblearn[torch]
pip install jupyterlab
```

Download the datasets (this might take a few minutes):
```
python scripts/datasets.py
```
The data will be stored in `./datasets/download`; the path can be customized with the environment variable `CBLEARN_DATA`.


Running matlab:
singularity run --env MLM_LICENSE_FILE=27000@matlab-campus.uni-tuebingen.de docker://mathworks/matlab:r2022a

docker run -it --rm -p 8888:8888 -e MLM_LICENSE_FILE=27000@matlab-campus.uni-tuebingen.de --shm-size=512M mathworks/matlab:r2022a 

## Plotting

Plots that visualize the datasets and the comparison's results, like the ones in the paper, are generated with jupyter notebooks.

Start jupyter `jupyter lab .`, and then run the following notebooks:

* `scripts/plot_datasets.ipynb` ![Datasets plot](plots/datasets.png)


## Libraries and Algorithms: 

**R-language** `R embedding.R <algo> <dataset> <result>`

* [MLDS](https://cran.r-project.org/web/packages/MLDS/index.html): MLDS algorithm
* [loe](https://cran.r-project.org/web/packages/loe/index.html): SOE algorithm

**Matlab** `matlab embedding.m -r "embedding <algo> <dataset> <result>"`

* [STE](https://lvdmaaten.github.io/ste/Stochastic_Triplet_Embedding.html): CKL[-K], GNMDS[-K], STE[-K], and tSTE algorithms.

**Python**


* [cblearn](https://github.com/dekuenstle/cblearn): MLDS, CKL-X, GNMDS-X, SOE, STE-X, tSTE CKL-GPU[-K], FORTE-GPU[-K], GNMDS-GPU[-K], SOE-GPU, STE-GPU, tSTE-GPU

how about ??? https://github.com/gcr/cython_tste
## Dependencies

### R Dependencies

* [docopt.R](https://github.com/docopt/docopt.R): Command line interface
* [rjson](https://cran.r-project.org/web/packages/rjson/index.html): JSON loading
* [MLDS](https://cran.r-project.org/web/packages/MLDS/index.html): MLDS Algorithm
* [loe](https://cran.r-project.org/web/packages/loe/index.html): SOE Algorithm
  
If you don't run the scripts with containers, you can manually install 
these dependencies to your local R instance with `install.packages(...)`.
