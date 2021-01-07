library(scales)
library(tidyverse)
library(loop.estimator)
library(kableExtra)
source('code/loop_ols.R')
source('code/loop_ext.R')

covNames <- c(
    "Prior.Problem.Count",
    "Prior.Percent.Correct",
    "Prior.Assignments.Assigned",
    "Prior.Percent.Completion",
    "Prior.Class.Percent.Completion",
    "Prior.Homework.Assigned",
    "Prior.Homework.Percent.Completion",
    "Prior.Class.Homework.Percent.Completion",
    "male",
    "unknownGender")#)


## ----loadData,include=FALSE,cache=FALSE----------------------------------
source('code/dataPrep.r')

## ----analysisFunctions,include=FALSE,cache=FALSE-------------------------
source('code/analysisFunctions.r')

## ----runAnalysis,include=FALSE,cache=TRUE--------------------------------
fullres <- sapply(levels(dat$problem_set),full,dat=dat,covNames=covNames,simplify=FALSE)

rnk <- rank(sapply(fullres,function(x) x['simpDiff','se']))
names(fullres) <- as.character(rnk)#LETTERS[rnk]

for(i in 1:length(fullres))
    attr(fullres[[i]],'psid') <- levels(dat$problem_set)[i]

save(fullres,file='results/fullres.RData')

#load('results/fullres.RData')

dat$ps <- rnk[as.character(dat$problem_set)]

source('code/combineFigures.r')
