# Replication code and data for _Precise Unbiased Estimation in Randomized Experiments using Auxiliary Observational Data_

https://arxiv.org/abs/2105.03529


## Replicating the Main Results in the Paper

The data for replicating the results in the main body of the paper are in two .csv files in the data/folder. 
These include imputations from the remnant in the column `p_complete`

First, clone the repository. Then the results reported in the manuscript can be replicated in `R` by running the following:

```
knitr::knit('MainResults.Rnw')
```
from the repo's directory. This will create the file `MainResults.tex` which you can then compile into a pdf with `pdflatex`

## Replicating Simulation Results

To replicate simulation results (in the appendix) run `R` from within the `sim code` directory.
The `loop.estimator` package needs to be installed from local files in the `loop.estimator` sub-directory, or simply `source` in all of the `.R` files:

```
for(file in list.files('loop.estimator/R')) if(endsWith(file,'.R')) source(paste0('loop.estimator/R/',file))
```
Then, to run the simulations, and create the plots in the appendix, use the following syntax (the `rmarkdown` package needs to be installed first)

```
rmarkdown::render('Plot Results.Rmd')
```

## Replicating Deep Learning Model in the Remnant

The imputations used in the paper's main results (and labeled `p_complete` in the datafile) were the result of two different deep learning models fit using historical data from ASSISTments, and then applied to user data from the experiments--the 22 experiments in `data/updated_exp_predictions.csv` (Study 1) and the 11 experiments in `data/newExperiments.csv`. This was all done in python.

Code for replicating this segment of the analysis is documented in two Jupyter Notebooks, `Python Code/Study_1/Study1_Model.ipynb` and `Python Code/Study_2/Study2_Model.ipynb`. The requisite data files are stored in .zip files in `Python Code/Study_1/resources/resources.zip` and `Python Code/Study_2/resources/resources.zip`; code for extracting these files is incorporated into the Jupyter notebooks. 
