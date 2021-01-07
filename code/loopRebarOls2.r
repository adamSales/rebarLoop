library(scales)
library(ggplot2)
library(dplyr)
#library(loop.estimator)
source('code/loop_ols.R')

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

## try loop once:
## lp1 <- loop(dat$complete[dat$problem_set==226210],
##  dat$treatment[dat$problem_set==226210],
##  covs[dat$problem_set==226210,])#,reg=FALSE)



res <- NULL
for(ps in unique(dat$problem_set)){
    print(ps)
    res <- rbind(res,
                 with(subset(dat,problem_set==ps),
                      c(ps=as.numeric(ps),
                        regEst=loop_ols(complete,treatment,1),
                        rebarEst=loop_ols(residual,treatment,1),
                        justCovs=loop_ols(complete,treatment,covs[dat$problem_set==ps,]),
                        rebarLoop=loop_ols(complete,treatment,p_complete),
                        predCovs=loop_ols(complete,treatment,
                            cbind(p_complete,covs[dat$problem_set==ps,]))
                        )
                      )
                 )
}

save(res,file='loopOLSresults.RData')

colnames(res) <- gsub(2,'SE',colnames(res))
colnames(res) <- gsub(1,'',colnames(res))

improvement <- apply(res[,grep('SE',colnames(res))],2,function(x) 1-sqrt(x/res[,'regEstSE']))

plot(sort(improvement[,'rebarLoopSE']))

### case # 14 plot for CODE
plotDat <- data.frame(analysis=factor(c('Unadj.','Aggregate\nCovariates','Rebar+\nLOOP','Rebar+\nLOOP+\nCovariates'),levels=c('Unadj.','Aggregate\nCovariates','Rebar+\nLOOP','Rebar+\nLOOP+\nCovariates')),
                      est=res[14,c('regEst','justCovs','rebarLoop','predCovsSE')],
                      se=sqrt(res[14,c('regEstSE','justCovsSE','rebarLoopSE','predCovsSE')]))

ggplot(plotDat,aes(analysis,est))+geom_point(size=4)+geom_errorbar(aes(ymin=est-2*se,ymax=est+2*se),width=0,size=3)+
    ylab('Treatment Effect')+xlab(NULL)+geom_hline(yintercept=0,linetype='dotted',size=2)+
    theme(axis.title=element_text(size=18),axis.text.x=element_text(size=18))


plotDat <- data.frame(x=1:22,y=sort(improvement[,'rebarLoopSE']))
ggplot(plotDat,aes(x,y))+geom_point(size=4)+geom_hline(yintercept=0,linetype='dotted',size=2)+
    ylab('% SE Improvement')+xlab(NULL)+
    theme(axis.title=element_text(size=18),axis.text.x=element_blank(),axis.text.y=element_text(size=14))+
    scale_y_continuous(labels=percent)
