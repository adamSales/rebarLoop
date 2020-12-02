library(scales)
library(tidyverse)
library(loop.estimator)
source('code/loop_ols2.R')
source('code/loop_ext.R')

## ----loadData,include=FALSE,cache=FALSE----------------------------------
source('code/dataPrep.r')

## ----analysisFunctions,include=FALSE,cache=FALSE-------------------------
source('code/analysisFunctions.r')

## ----runAnalysis,include=FALSE,cache=TRUE--------------------------------
#fullres <- sapply(levels(dat$problem_set),full,simplify=FALSE)

#save(fullres,file='results/fullres.RData')

load('results/fullres.RData')
rnk <- rank(sapply(fullres,function(x) x['simpDiff','se']))
names(fullres) <- LETTERS[rnk]
dat$ps <- LETTERS[rnk[as.character(dat$problem_set)]]

### a little experiment
## n=50
s1.50 <- s3.50 <- matrix(nrow=1000,ncol=22)
ses.50 <- array(dim=c(1000,4,22),dimnames=list(NULL,c('simpDiff','strat1','strat3','justCovs'),levels(dat$problem_set)))


for(i in 1:1000){
  res100 <- sapply(levels(dat$problem_set),full,n=50,methods=c('simpDiff','strat1','strat3','justCovs'),
    simplify=FALSE)
  ses.50[i,,] <- sapply(res100,function(x) x[,'se'])
}

s1.50 <- apply(ses.50,1,function(x) (x['justCovs',]/x['strat1',])^2)
s3.50 <- apply(ses.50,1,function(x) (x['justCovs',]/x['strat3',])^2)

save(ses.50,s1.50,s3.50,file='results/smallSampleReplication50.RData')

pdf('figure/smallSample50.pdf')
hist(s1.50,main='V(LOOP)/V(Strat 1) n=50; pooled',xlab=paste0(round(mean(s1.50>=1)*100,1),'% >=1; range=[',round(min(s1.50),2),',',round(max(s1.50),2),']'))

rownames(s1.50) <- names(fullres)
boxplot(t(s1.50),main='V(LOOP)/V(Strat 1) n=50')
text(1:22,4,apply(s1.50,1,function(x) round(mean(x>=1)*100)),cex=0.8)
text(11,4.2,'% >= 1:')
abline(h=1,lty='dotted')

hist(s3.50,main='V(LOOP)/V(Strat 3) n=50; pooled',
  xlab=paste0(round(mean(s3.50>=1,na.rm=TRUE)*100,1),'% >=1; range=[',round(min(s3.50,na.rm=TRUE),2),',',round(max(s3.50,na.rm=TRUE),2),']'))

rownames(s3.50) <- names(fullres)
boxplot(t(s3.50),main='V(LOOP)/V(Strat 3) n=50')
text(1:22,4,apply(s3.50,1,function(x) round(mean(x>=1,na.rm=TRUE)*100)))
text(11,4.2,'% >= 1:')
abline(h=1,lty='dotted')

dev.off()

## n=100
s1.100 <- s3.100 <- matrix(nrow=1000,ncol=22)
ses.100 <- array(dim=c(1000,4,22),dimnames=list(NULL,c('simpDiff','strat1','strat3','justCovs'),levels(dat$problem_set)))


for(i in 1:1000){
  res100 <- sapply(levels(dat$problem_set),full,n=100,methods=c('simpDiff','strat1','strat3','justCovs'),
    simplify=FALSE)
  ses.100[i,,] <- sapply(res100,function(x) x[,'se'])
}

s1.100 <- apply(ses.100,1,function(x) (x['justCovs',]/x['strat1',])^2)
s3.100 <- apply(ses.100,1,function(x) (x['justCovs',]/x['strat3',])^2)

save(ses.100,s1.100,s3.100,file='results/smallSampleReplication100.RData')

pdf('figure/smallSample100.pdf')
hist(s1.100,main='V(LOOP)/V(Strat 1) n=100; pooled',xlab=paste0(round(mean(s1.100>=1)*100,1),'% >=1; range=[',round(min(s1.100),2),',',round(max(s1.100),2),']'))

rownames(s1.100) <- names(fullres)
boxplot(t(s1.100),main='V(LOOP)/V(Strat 1) n=100')
text(1:22,2,apply(s1.100,1,function(x) round(mean(x>=1)*100)),cex=0.8)
text(11,2.05,'% >= 1:')
abline(h=1,lty='dotted')

hist(s3.100,main='V(LOOP)/V(Strat 3) n=100; pooled',
  xlab=paste0(round(mean(s3.100>=1,na.rm=TRUE)*100,1),'% >=1; range=[',round(min(s3.100,na.rm=TRUE),2),',',round(max(s3.100,na.rm=TRUE),2),']'))

rownames(s3.100) <- names(fullres)
boxplot(t(s3.100),main='V(LOOP)/V(Strat 3) n=100')
text(1:22,2,apply(s3.100,1,function(x) round(mean(x>=1,na.rm=TRUE)*100)))
text(11,2.05,'% >= 1:')
abline(h=1,lty='dotted')

dev.off()




res100 <- sapply(levels(dat$problem_set),full,n=100,methods=c('simpDiff','strat1','strat3','justCovs'),
  simplify=FALSE))



## ----loadResults,include=FALSE,cache=FALSE-------------------------------

## load('results/res100.RData')
## names(res100) <- LETTERS[rnk]


pd <- do.call('rbind',
  lapply(names(fullres),
    function(x) cbind(as.data.frame(fullres[[x]]),
      method=factor(rownames(fullres[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x)))%>%
   select(-improvement)%>%
   rename(experiment=rnk)%>%
   filter(method!='strat2')%>%
   mutate(method=fct_recode(.$method,ReLoopStar='strat1',ReLoop='strat3',loop='justCovs'))

justSEs <- sapply(fullres,function(x) x[,'se'])



justImp <- round(sapply(fullres,function(x) x[,'improvement'])*100)

## ----sampleSize,results='asis',echo=FALSE--------------------------------
ssTab <- table(dat$treatment,dat$ps)
cat('Experiment &',paste(LETTERS[1:11],collapse='&'),'\\\\\n')
cat('\\hline\n')
cat('$n_C$ &',paste(ssTab['0',LETTERS[1:11]],collapse='&'),'\\\\\n')
cat('$n_T$ &',paste(ssTab['1',LETTERS[1:11]],collapse='&'),'\\\\\n')
cat('\\hline\n')
cat('Experiment &',paste(LETTERS[12:22],collapse='&'),'\\\\\n')
cat('\\hline\n')
cat('$n_C$ &',paste(ssTab['0',LETTERS[12:22]],collapse='&'),'\\\\\n')
cat('$n_T$ &',paste(ssTab['1',LETTERS[12:22]],collapse='&'),'\\\\\n')

## ----makeGraphics,include=FALSE------------------------------------------
source('code/graphics.r')

## ----rebar,include=FALSE-------------------------------------------------
best <- colnames(justImp)[which.max(justImp['rebar',])]
badRebar1 <- colnames(justImp)[which.min(justImp['rebar',])]
ji <- justImp[,-which.min(justImp['rebar',])]
badRebar2 <- colnames(ji)[which.min(ji['rebar',])]

## ----badReLOOP,include=FALSE---------------------------------------------
badReLOOP1 <- -justImp['strat1',justSEs['strat1',]>justSEs['simpDiff',]]
badReLOOP <- -justImp['strat3',justSEs['strat3',]>justSEs['simpDiff',]]

## ----reloop,include=FALSE------------------------------------------------
bestL <- colnames(justImp)[which.max(justImp['justCovs',])]
stopifnot(sum(justSEs['justCovs',]>justSEs['simpDiff',])==4)

