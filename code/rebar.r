library(sandwich)
library(ggplot2)

dat <- read.csv('updated_exp_predictions.csv')
dat <- subset(dat,ExperiencedCondition)
dat$problem_set <- as.factor(dat$problem_set)

### analysis for "complete"
dat <- droplevels(dat[!is.na(dat$p_complete),])

### sample sizes?
table(dat$problem_set,dat$condition)


### how good are the predictions?
dat$completeR <- dat$complete-dat$p_complete

(summary(m1 <- lm(complete~problem_set,data=dat,subset=condition=='C')))
(summary(m1 <- lm(completeR~problem_set,data=dat,subset=condition=='C')))


est <- function(Y,Z){
    mean(Y[Z=='E'])-mean(Y[Z=='C'])
}
se <- function(Y,Z){
    n <- length(Y)
    n1 <- sum(Z=='E')
    n0 <- n-n1
    sqrt(n/n1/n0)*sd(Y[Z=='C'])
}

res <- NULL
for(ps in unique(dat$problem_set))
    res <- rbind(res,
                 with(subset(dat,problem_set==ps),
                      c(ps=as.numeric(ps),
                        regEst=est(complete,condition),
                        regSE=se(complete,condition),
                        rebarEst=est(completeR,condition),
                        rebarSE=se(completeR,condition))))


#qplot(res[,'regSE'],res[,'rebarSE'],xlim=range(c(res[,'regSE'],res[,'rebarSE'])),ylim=range(c(res[,'regSE'],res[,'rebarSE'])),xlab='Unadjusted Standard Errors',ylab='Rebar Standard Errors')+geom_abline(intercept=0,slope=1,linetype='dotted')+coord_fixed()

#qplot(1:22,sort((res[,'rebarSE']-res[,'regSE'])/res[,'regSE'])*100,ylab='% Change in SE')+geom_hline(yintercept=0,linetype='dotted')+scale_x_continuous(NULL)+scale_y_continuous(breaks=seq(-40,10,10),labels=paste0(seq(-40,10,10),'%'))


plotDat <- as.data.frame(rbind(res[order(res[,2]),2:3],res[order(res[,2]),c(4,5)])*100)
plotDat$type=factor(rep(c('Usual','Rebar'),each=nrow(res)),levels=c('Usual','Rebar'))
plotDat$x <- c(1:22,1:22+0.2)
names(plotDat)=c('est','se','type','x')

ggplot(plotDat,aes(x,est,fill=type,color=type))+geom_point()+geom_errorbar(aes(ymin=est-2*se,ymax=est+2*se))+geom_hline(yintercept=0,linetype='dotted')+ylab('Estimated Treatment Effect\n(Percentage Point)')+xlab('Experiment')+#scale_x_continuous(breaks=seq(1.1,22.1,1),labels=1:22)+
    labs(color=NULL,fill=NULL)+
         theme(legend.position='bottom',text=element_text(size=7.5),
              axis.title.x=element_blank(),
              axis.text.x=element_blank(),
              axis.ticks.x=element_blank())#theme(legend.position='top',text=element_text(size=7.5))
ggsave('estEff1.jpg',width=3,height=3)

cors <- sapply(unique(dat$problem_set),function(ps) with(subset(dat,problem_set==ps),cor(p_complete,complete)))
predR2 <- sapply(unique(dat$problem_set),function(ps) with(subset(dat,problem_set==ps & condition=='C'),1-sum((p_complete-complete)^2)/sum((complete-mean(complete))^2)))
bias <- sapply(unique(dat$problem_set),function(ps) with(subset(dat,problem_set==ps & condition=='C'),mean(p_complete-complete)^2))
mses <- sapply(unique(dat$problem_set),function(ps) with(subset(dat,problem_set==ps),mean((p_complete-complete)^2)))
seDiff <- (res[,'regSE']-res[,'rebarSE'])/res[,'regSE']
qplot(predR2*100,seDiff*100,xlab=expression(paste('Prediction ',R^2)),ylab='% SE Reduction')+geom_hline(yintercept=0,linetype='dotted')+scale_y_continuous(breaks=seq(-10,50,10),labels=paste0(seq(-10,50,10),'%'))+theme(text=element_text(size=7.5))
ggsave('corVsSE.jpg',width=3,height=3)


### look at per-PS variables
dat$grade <- dat$Class.Grade
dat$grade[dat$Class.Grade=='Freshman'] <- 9
dat$grade[dat$Class.Grade=='Sophomore'] <- 10
dat$grade[dat$Class.Grade=='Junior'] <- 11
dat$grade[dat$Class.Grade=='Senior'] <- 12
dat$grade[dat$Class.Grade=='N/A'] <- NA
dat$grade <- as.numeric(dat$grade)

library(dplyr)

psDat <- dat%>%group_by(problem_set)%>%summarize(n=n(),grade=mean(grade,na.rm=TRUE),
                                                 fem=mean(Guessed.Gender==levels(Guessed.Gender)[2],na.rm=TRUE),
                                                 noGen=mean(Guessed.Gender==levels(Guessed.Gender)[3],na.rm=TRUE),comp=mean(p_complete,na.rm=TRUE),
                                                 invMast=mean(p_inverted_mastery,na.rm=TRUE),nprob=mean(Prior.Problem.Count,na.rm=TRUE),
                                                 priorCorr=mean(Prior.Percent.Correct,na.rm=TRUE),complete=mean(complete,na.rm=TRUE))
psDat$ratio <- res[,'rebarSE']/res[,'regSE']
as.data.frame(psDat%>%arrange(ratio))

par(mfrow=c(3,3))
for(i in 2:10) plot(psDat[[i]],psDat$ratio,main=names(psDat)[i])

badps <- psDat$problem_set[psDat$ratio>1]

psDat$imp <- 1-psDat$ratio

qplot(n,imp*100,xlab='Sample Size',ylab='% SE Reduction',data=psDat)+geom_hline(yintercept=0,linetype='dotted')+scale_y_continuous(breaks=seq(-10,50,10),labels=paste0(seq(-10,50,10),'%'))+theme(text=element_text(size=7.5))+scale_x_continuous(breaks=seq(100,2000,100))#+geom_smooth(method='lm',se=FALSE)
ggsave('sampleSize.jpg',width=3,height=3)

## res2 <- NULL
## for(ps in unique(dat$problem_set)){
##     summary(mod1 <- lm(complete~condition+Prior.Percent.Correct+Prior.Percent.Completion,subset=problem_set==ps,data=dat))
##     se1 <- sqrt(vcovHC(mod1,'HC2')[2,2])
##     mod2 <- lm(completeR~condition+Prior.Percent.Correct+Prior.Percent.Completion,subset=problem_set==ps,data=dat)
##     se2 <- sqrt(vcovHC(mod2,'HC2')[2,2])
##     res2 <- rbind(res2,c(ps=as.numeric(ps),regEst=coef(mod1)[2],regSE=se1,rebarEst=coef(mod2)[2],rebarSE=se2))
## }

## plotDat <- as.data.frame(rbind(res2[order(res2[,2]),2:3],res2[order(res2[,2]),c(4,5)])*100)
## plotDat$type=factor(rep(c('Usual','Rebar'),each=nrow(res2)),levels=c('Usual','Rebar'))
## plotDat$x <- c(1:22,1:22+0.2)
## names(plotDat)=c('est','se','type','x')


## ggplot(plotDat,aes(x,est,fill=type,color=type))+geom_point()+geom_errorbar(aes(ymin=est-2*se,ymax=est+2*se))+geom_hline(yintercept=0,linetype='dotted')+ylab('Estimated Treatment Effect\n(Percentage Point)')+xlab('Experiment')+#+scale_x_continuous(NULL) +#breaks=seq(1.1,22.1,1),labels=1:22)+
##     labs(color=NULL,fill=NULL)+
##         theme(legend.position='top',#text=element_text(size=7.5),
##               axis.title.x=element_blank(),
##               axis.text.x=element_blank(),
##               axis.ticks.x=element_blank())
## ggsave('estEff2.jpg')#,width=3,height=3)

## xcors <- sapply(unique(dat$problem_set),function(ps) with(subset(dat,problem_set==ps),cor(p_complete,complete)))
## seDiff <- (res2[,'regSE']-res2[,'rebarSE'])/res2[,'regSE']
## qplot(cors,seDiff*100,xlab=expression(paste('Cor(',hat(Y),',',Y,')')),ylab='% Improvement')+geom_hline(yintercept=0,linetype='dotted')+scale_y_continuous(breaks=seq(-10,50,10),labels=paste0(seq(-10,50,10),'%'))+theme(text=element_text(size=7.5))
## ggsave('corVsSE2.jpg',width=3,height=3)


