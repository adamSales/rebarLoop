library(scales)
library(ggplot2)
library(dplyr)
library(loop.estimator)
source('code/loop_ols.R')
source('code/loop_ext.R')

dat <- read.csv('updated_exp_predictions.csv')

dat$inverted_mastery <- ifelse(dat$complete==1,1/dat$problem_count,0)

## outcome is 'complete'
## prediction is 'p_complete'

## only keep those subjects whom treatment could possibly affect:
dat <- subset(dat,ExperiencedCondition)

## this is the list of experiments
dat$problem_set <- as.factor(dat$problem_set)

### analysis for "inverted_mastery"; drop subjects for whom "predicted" inverted_mastery is NA
dat <- droplevels(dat[!is.na(dat$p_inverted_mastery),])

### sample sizes?
table(dat$problem_set,dat$condition)

### data frame of covariates from the dataset
covs <- subset(dat,select=c(
                       Prior.Problem.Count,
                       Prior.Percent.Correct,
                       Prior.Assignments.Assigned,
                       Prior.Percent.Completion,
                       Prior.Class.Percent.Completion,
                       Prior.Homework.Assigned,
                       Prior.Homework.Percent.Completion,
                       Prior.Class.Homework.Percent.Completion,
                       Guessed.Gender))
## excluded "birthyear" cuz it's weird--some students obv messing around

### mean imputation for covariates:
covs <- as.data.frame(lapply(covs,function(x){ x[is.na(x)] <- mean(x,na.rm=TRUE); x}))
covs$male <- covs$Guessed.Gender=='Male'
covs$unknownGender <- covs$Guessed.Gender=='Uknown'
covs$Guessed.Gender <- NULL

dat$treatment <- ifelse(dat$condition=='E',1,0)

dat$residual <- dat$inverted_mastery-dat$p_inverted_mastery

### simple difference estimator
### Neyman variance estimate
simpDiff <- function(datPS,covsPS,outcome='inverted_mastery'){
  Y <- datPS[[outcome]]
  Z <- datPS$treatment
  est <- mean(Y[Z==1])- mean(Y[Z==0])
  Vhat <- var(Y[Z==1])/sum(Z==1)+var(Y[Z==0])/sum(Z==0)
  return(c(tauhat=est,varhat=Vhat))
}

### rebar estimate: simple difference on residuals
rebar <- function(datPS,covsPS)
  simpDiff(datPS,covsPS,outcome='residual')

## OLS LOOP with deep learning predictions as the only covariate
strat1 <- function(datPS,covsPS)
    with(datPS,
         loop_ols(inverted_mastery,treatment,p_inverted_mastery))

## RF LOOP with deep learning predictions alongside other covariates
strat2 <- function(datPS,covsPS)
    with(datPS,
         loop(inverted_mastery,treatment,cbind(p_inverted_mastery,covsPS)))

## Combine RF LOOP with covariates with OLS LOOP with deep learning predictions
strat3 <- function(datPS,covsPS)
    with(datPS,
         loop_ext(inverted_mastery,treatment,covsPS,extm=p_inverted_mastery))

## RF LOOP with covariates, no deep learning predictions
justCovs <- function(datPS,covsPS)
    with(datPS,
         loop(inverted_mastery,treatment,covsPS))

## all the estimators, for a particular problem set
full <- function(ps){
    datPS <- dat[dat$problem_set==ps,]
    covsPS <- covs[dat$problem_set==ps,]

    res <- sapply(c('simpDiff','rebar','strat1','strat2','strat3','justCovs'),
                  function(FUN){
                      fun <- get(FUN)
                      fun(datPS,covsPS)
                  },simplify=FALSE)
    res <- do.call('rbind',res)
    res[,2] <- sqrt(res[,2])  ## standard error, not variance
    colnames(res) <- c('est','se')
    res <- cbind(res,improvement=1-res[,'se']/res['simpDiff','se'])
    res
}

fullres <- sapply(levels(dat$problem_set),full,simplify=FALSE)

## save(fullres,file='results/fullres.RData')

## ggplot(dat,aes(p_inverted_mastery,inverted_mastery))+geom_point()+facet_wrap(~problem_set)

## sort(sapply(fullres,function(x) x['strat3','improvement']))
