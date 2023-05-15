# Replication code for _Precise Unbiased Estimation in Randomized Experiments using Auxiliary Observational Data_

This contains the code to reproduce the results in the manuscript _Precise Unbiased Estimation in Randomized Experiments using Auxiliary Observational Data_ (Gagnon-Bartsch and Sales, et al).  

## Obtaining the code and data

* Although not required, the code here is intended to be run in Docker.  
  * A docker image containing this code can be found at <https://osf.io/d9ujq/>.  
  * To use the docker image, first download it, then install it and run it with  
    * `docker load precise2023.tar.gz`  
    * `docker run -it --rm -p 127.0.0.1:80:80 precise2023`  
  * For reference, the Dockerfile is in the `docker` directory.  
* **The data files are not included here.**
  * Data files are available at <https://osf.io/j6esa/>.
  * Note that the data can be downloaded, but not redistributed.
  * Likewise, the data is not included in the docker image.
  * The data is automatically downloaded with `data/download.Rmd`

## Notes on the code

**All of the analyses can be run by typing `make` in the root directory.**  (This also first downloads the data.)

* The `analysis` folder contains code notebooks in markdown format that can be run, e.g., in Jupyter.
  * `01_remnant_model_1.md`: Fits the auxiliary data model for the first 22 experiments.
  * `02_remnant_model_2.md`: Fits the auxiliary data model for the other 11 experiments.
  * `03_assemble_data.Rmd`: Assembles the full 33-experiment dataset, including the predictions from the models above.
  * `04_estimate.Rmd`: Runs the main analyses of the paper and saves the output.  
  * `05_outputs.Rmd`: Produces the plots, tables, and other numerical results that are in the paper.
  * `06_appendixD.Rmd`: Reproduces the figures in Appendix D.
  * `scripts`: Additional python scripts used by the remnant models.
* The `data/original` folder should contain the raw data downloaded from <https://osf.io/j6esa/>.
* The `docker` folder only contains files necessary to build the docker image.
* The `output` folder contains the output (figures, tables, etc.) from running the code.  
* The `package` folder contains an R package that implements the statistical methods from the paper.
