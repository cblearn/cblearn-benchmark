# cblearn-benchmark

This repository contains a small empirical comparison of algorithm implementations in [cblear(https://github.com/dekuenstle/cblearn) 
with each other and with implementations in different libraries. 

At the moment, only ordinal embedding algorithms are evaluated. 


## Install
singularity run --env MLM_LICENSE_FILE=27000@matlab-campus.uni-tuebingen.de docker://mathworks/matlab:r2022a

docker run -it --rm -p 8888:8888 -e MLM_LICENSE_FILE=27000@matlab-campus.uni-tuebingen.de --shm-size=512M mathworks/matlab:r2022a 

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