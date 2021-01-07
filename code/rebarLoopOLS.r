library(ggplot2)
library(dplyr)
#library(loop.estimator)
source('code/loop_ols.r')

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


## try loop once:
## lp1 <- loop(dat$complete[dat$problem_set==226210],
##  dat$treatment[dat$problem_set==226210],
##  covs[dat$problem_set==226210,])#,reg=FALSE)


## estimate and varhat
est <- function(Y,Z,yhat){
    if(!missing(yhat))
        Y <- Y-yhat
    mean(Y[Z==1])-mean(Y[Z==0])
}

varhat <- function(Y,Z,yhat){
    n_t <- sum(Z)
    n_c <- sum(1-Z)
    p <- 1/2
    if(missing(yhat)){ ### this is for the standard difference in means
        M_t <- var(Y[Z==1])
        M_c <- var(Y[Z==0])
    } else{
        M_t <- sum((yhat[Z==1]-Y[Z==1])^2)/n_t
        M_c <- sum((yhat[Z==0]-Y[Z==0])^2)/n_c
    }

    1/(n_t + n_c) * ((1 - p)/p * M_t + p/(1 - p) * M_c +
                         2 * sqrt(M_t * M_c))
}

res <- NULL
for(ps in unique(dat$problem_set)){
    print(ps)
    res <- rbind(res,
                 with(subset(dat,problem_set==ps),
                      c(ps=as.numeric(ps),
                        regEst=loop_ols(complete,treatment,1),#est(complete,treatment),
                        #regVARHAT=varhat(complete,treatment),
                        #rebarEst=est(complete,treatment,p_complete),
                        #rebarVARHAT=varhat(complete,treatment,p_complete),
                        justCovs=loop_ols(complete,treatment,covs[dat$problem_set==ps,]),
                        #justPred=loop_ols(complete,treatment,cbind(p_complete)),
                        predCovs=loop_ols(complete,treatment,
                            cbind(p_complete,covs[dat$problem_set==ps,]))#,
                        #predRF=loop(complete,treatment,cbind(p_complete))
                        )
                      )
                 )
}

## plot results
seCols <- c('regVARHAT','rebarVARHAT','justCovs2','justPred2','predCovs2')
seMax <- max(res[,seCols])
pdf('compareSEsOLS.pdf')
for(v1 in 1:(length(seCols)-1))
    for(v2 in (v1+1):length(seCols)){
        plot(res[,seCols[v1]],res[,seCols[v2]],pch=16,ylim=c(0,seMax),xlim=c(0,seMax),
             xlab=seCols[v1],ylab=seCols[v2],asp=1)
        abline(0,1)
    }
dev.off()

plotDat <- as.data.frame(rbind(res[order(res[,2]),2:3],res[order(res[,2]),c(4,5)],res[order(res[,2]),c(6,7)]))
plotDat$type=factor(rep(c('Mean Difference','Cov\'s','Cov\'s+Predictions'),each=nrow(res)),levels=c('Mean Difference','Covariates','Covariates+Predictions'))

plotDat$x <- c(1:22,1:22+0.2,1:22+0.4)
names(plotDat)=c('est','se','type','x')
plotDat$se <- sqrt(plotDat$se)

ggplot(plotDat,aes(x,est,fill=type,color=type))+geom_point()+geom_errorbar(aes(ymin=est-2*se,ymax=est+2*se),width=0)+geom_hline(yintercept=0,linetype='dotted')+ylab('Estimated Treatment Effect\n(Percentage Point)')+xlab('Experiment')+#scale_x_continuous(breaks=seq(1.1,22.1,1),labels=1:22)+
    labs(color=NULL,fill=NULL)+
         theme(legend.position='bottom',text=element_text(size=7.5),
              axis.title.x=element_blank(),
              axis.text.x=element_blank(),
              axis.ticks.x=element_blank())#theme(legend.position='top',text=element_text(size=7.5))
ggsave('estEff1.jpg',width=3,height=3)

plot(sqrt(res[,'regEst2']),sqrt(res[,'predCovs2']/res[,'regEst2']))

qplot(1:22,sort(sqrt(res[,'predCovs2']/res[,'regEst2'])),xlab=NULL,ylab='Aux. Data SE/Usual SE',geom='point')+geom_hline(yintercept=1,linetype='dotted')
