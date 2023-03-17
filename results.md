# Results

Here we show the lastest result plots stored in this repository. 


# Dataset overview


![](./plots/datasets.png) ![](./plots/error-per-dataset_cblearn.png  ) 

Datapoints in the boxplot correspond to runs of different cpu embedding algorithms, implemented in *cblearn*. 

# Algorithm comparison by subgroups

The shown errors and runtimes are relative to the average performance of the shown subgroup.
Datapoints are the previously shown datasets. 

| Error      | Time        |
|------------|-------------|
| ![](./plots/deltaerror-per-algorithm_cblearn-all.png ) | ![](./plots/deltatime-per-algorithm_cblearn-all.png )  |
| ![](./plots/deltaerror-per-algorithm_library.png ) | ![](./plots/deltatime-per-algorithm_library.png ) |
| ![](./plots/deltaerror-per-algorithm_gpu.png ) | ![](./plots/deltatime-per-algorithm_gpu.png ) |
| ![](./plots/error-per-dataset-algorithm_cblearn.png ) | ![](./plots/time-per-dataset-algorithm_cblearn.png  ) |


# Performance per number of triplets 

Here we see, that the used datasets are still too small for the GPU to be beneficial. 

![](./plots/time-per-triplets_cblearn.png ) ![](./plots/time-per-triplets_gpu.png )




