# Replication code fo _Precise Unbiased Estimation in Randomized Experiments using Auxiliary Observational Data_

This contains the code to reproduce the results in the manuscript _Precise Unbiased Estimation in Randomized Experiments using Auxiliary Observational Data_.  A few notes:

* The code here is intended to be run in Docker, although this is not required.  The Dockerfile is in the `docker` directory.  

* The data files are not included here but are available at 

https://osf.io/j6esa/

and can be automatically downloaded with `data/download.Rmd`.

* The `analysis` folder contains code notebooks in markdown format that can be run, e.g., in Jupyter.
  * `01_remnant_model_1.md`: Fits the auxiliary data model for the first 22 experiments.
  * `02_remnant_model_2.md`: Fits the auxiliary data model for the other 11 experiments.
  * `03_assemble_data.Rmd`: Assembles the full 33-experiment dataset, including the predictions from the models above.
  * `04_estimate.Rmd`: Runs the main analyses of the paper and saves the output.  Note that the statistical methods are implemented in an R package in the `package` directory.
  * `05_outputs.Rmd`: Produces the plots, tables, and other numerical results that are in the paper.
  * `06_appendixD.Rmd`: Reproduces the figures in Appendix D.
  * `scripts`: Additional python scripts used by the remnant models.
  
* All of the analyses can be run by typing `make` in the root directory.  (This also first downloads the data.)


