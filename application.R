library(scales)
library(tidyverse)
library(loop.estimator)
source('code/loop_ols.R')
source('code/loop_ext.R')

## ----loadData,include=FALSE,cache=FALSE----------------------------------
source('code/dataPrep.r')

## ----analysisFunctions,include=FALSE,cache=FALSE-------------------------
source('code/analysisFunctions.r')

## ----runAnalysis,include=FALSE,cache=TRUE--------------------------------
fullres <- sapply(levels(dat$problem_set),full,simplify=FALSE)

rnk <- rank(sapply(fullres,function(x) x['simpDiff','se']))
names(fullres) <- as.character(rnk)#LETTERS[rnk]

save(fullres,file='results/fullres.RData')

#load('results/fullres.RData')

dat$ps <- rnk[as.character(dat$problem_set)]

source('code/combineFigures.r')
