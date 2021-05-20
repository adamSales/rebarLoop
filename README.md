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
Then, to run the simulations, use the following syntax (the `rmarkdown` package needs to be installed first)

```
rmarkdown::render('sim_by_n.Rmd')
rmarkdown::render('sim_by_r2c.Rmd')
rmarkdown::render('sim_by_r2p.Rmd')
rmarkdown::render('Plot Results.Rmd')
```

## Replicating Deep Learning Model in the Remnant
