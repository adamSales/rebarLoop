library(scales)
library(ggplot2)
library(dplyr)
library(loop.estimator)
source('code/loop_ols.R')
source('code/loop_ext.R')

dat <- read.csv('updated_exp_predictions.csv')

## outcome is 'complete'
## prediction is 'p_complete'

## only keep those subjects whom treatment could possibly affect:
dat <- subset(dat,ExperiencedCondition)

## this is the list of experiments
dat$problem_set <- as.factor(dat$problem_set)

### analysis for "complete"; drop subjects for whom "predicted" complete is NA
dat <- droplevels(dat[!is.na(dat$p_complete),])

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

dat$residual <- dat$complete-dat$p_complete

## ## try loop once:
## ## lp1 <- loop(dat$complete[dat$problem_set==226210],
## ##  dat$treatment[dat$problem_set==226210],
## ##  covs[dat$problem_set==226210,])#,reg=FALSE)

## dat <- cbind(covs,select(dat,treatment,complete,p_complete))

simpDiff <- function(datPS,covsPS)
    with(datPS,
         loop_ols(complete,treatment,cbind(rep(1,nrow(datPS)))))

## rebar <- function(datPS,covsPS){
##     est <- mean(datPS$residual[datPS$treatment==1])-
##         mean(datPS$residual[datPS$treatment==0])
##     Ec <- mean(datPS$residual[datPS$treatment==0]^2)
##     Et <- mean(datPS$residual[datPS$treatment==1]^2)
##     Vhat <- 1/nrow(datPS)*(Ec+Et+2*sqrt(Ec*Et))
##     return(c(tauhat=est,varhat=Vhat))
## }

rebar <- function(datPS,covsPS)
    with(datPS,
         loop_ols(residual,treatment,cbind(rep(1,nrow(datPS)))))

strat1 <- function(datPS,covsPS)
    with(datPS,
         loop_ols(complete,treatment,p_complete))

strat2 <- function(datPS,covsPS)
    with(datPS,
         loop(complete,treatment,cbind(p_complete,covsPS)))

strat3 <- function(datPS,covsPS)
    with(datPS,
         loop_ext(complete,treatment,covsPS,extm=p_complete))

justCovs <- function(datPS,covsPS)
    with(datPS,
         loop(complete,treatment,covsPS))

full <- function(ps){
    datPS <- dat[dat$problem_set==ps,]
    covsPS <- covs[dat$problem_set==ps,]

    res <- sapply(c('simpDiff','rebar','strat1','strat2','strat3','justCovs'),
                  function(FUN){
                      fun <- get(FUN)
                      fun(datPS,covsPS)
                  },simplify=FALSE)
    res <- do.call('rbind',res)
    res[,2] <- sqrt(res[,2])
    colnames(res) <- c('est','se')
    res <- cbind(res,improvement=1-res[,'se']/res['simpDiff','se'])
    res
}

fullres <- sapply(levels(dat$problem_set),full,simplify=FALSE)

save(fullres,file='results/fullres.RData')

rnk <- rank(sapply(fullres,function(x) x['strat3','improvement']))

pd <- do.call('rbind',lapply(levels(dat$problem_set),
                             function(x) cbind(as.data.frame(fullres[[x]]),
                                               method=factor(rownames(fullres[[x]]),
                                                   levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
                                                   ps=x)))
pd$rnk <- LETTERS[rnk[pd$ps]]

pd <- droplevels(subset(pd,method%in%c('simpDiff','justCovs','rebar','strat3')))
levels(pd$method) <- c('Simple Difference','LOOP (Without Remnant)','Rebar','LOOP+Rebar')

ggplot(pd,aes(method,se,fill=method))+
    geom_col(position='dodge')+xlab(NULL)+
    theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          legend.position='top')+
        scale_fill_manual(values=subwayPalette[1:4],name=NULL)+
        facet_wrap(~rnk,scales="free_y")
ggsave('ses.pdf')


ggplot(filter(pd,method%in%c('Simple Difference','LOOP+Rebar')), aes(rnk,est,color=method))+
    geom_point(position=position_dodge(.4))+
        geom_errorbar(aes(ymin=est-2*se,ymax=est+2*se),position=position_dodge(.4),width=0)+
            geom_hline(yintercept=0,linetype='dotted')+
                theme(legend.position='top')+
                    labs(color=NULL,x=NULL,y='Treatment Effect')
ggsave('estimates.pdf')


