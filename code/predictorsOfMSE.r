library(sandwich)
library(ggplot2)
library(dplyr)
library(tidyr)

source('code/rebarLoop.r') ## gives 'res'= results

dat <- read.csv('updated_exp_predictions.csv')
dat$problem_set <- as.factor(dat$problem_set)
datFull <- dat

dat <- subset(dat,ExperiencedCondition)


### analysis for "complete"
dat <- droplevels(dat[!is.na(dat$p_complete),])

dat$grade <- as.character(dat$Class.Grade)
dat$grade[dat$Class.Grade=='"Freshmen"'] <- '9'
dat$grade[dat$Class.Grade=='"Sophomore"'] <- '10'
dat$grade[dat$Class.Grade=='"Junior"'] <- '11'
dat$grade[dat$Class.Grade=='"Senior"'] <- '12'
dat$grade[dat$Class.Grade=='"N/A"'] <- NA
dat$grade <- gsub('"','',dat$grade)
dat$grade <- as.numeric(dat$grade)



psDat <- aggregate(dat[,grep('Prior.',names(dat))],by=list(problem_set=dat$problem_set),FUN=mean,na.rm=TRUE)

psDat2 <- dat%>%group_by(problem_set)%>%
    summarize(sample.size=n(),
              avg.grade=mean(grade,na.rm=TRUE),
              grade.NA=mean(is.na(grade)),
              perc.fem=mean(Guessed.Gender==levels(Guessed.Gender)[2],na.rm=TRUE),
              perc.gender.unknown=mean(Guessed.Gender==levels(Guessed.Gender)[3],na.rm=TRUE),
              perc.missing.covariates=mean(is.na(Prior.Percent.Correct)),
              avg.Y=mean(complete,na.rm=TRUE),
              avg.predicted.Y=mean(p_complete,na.rm=TRUE),
              RMSE=sqrt(mean((complete-p_complete)^2,na.rm=TRUE)),
              R2=1-RMSE^2/var(complete,na.rm=TRUE),
              bias=mean(p_complete)-mean(complete))

psDat <- merge(psDat,psDat2)

for(ps in psDat$problem_set){
    #psDat$percent.experienced.condition=mean(datFull$ExperiencedCondition[datFull$problem_set==ps])
    psDat$percent.na.prediction[psDat$problem_set==ps] <- mean(is.na(datFull$p_complete)[datFull$problem_set==ps])
}

psDat$ps <- 1:nrow(psDat)

if(all(psDat$problem_set==res[,'ps'])){
    psDat$regVARHAT <- res[,'regVARHAT']
    psDat$improvement <- with(as.data.frame(res),1-justPred2/regVARHAT)
}

ddd <- psDat%>%select(-Prior.Correct.Count,-Prior.Completion.Count,-Prior.Homework.Count)%>%
    gather(Variable,Value,-c(problem_set,ps,improvement))

ggplot(ddd,aes(Value,improvement,label=ps))+geom_text()+geom_hline(yintercept=0,linetype='dotted')+ylab('Variance Improvement (1-V(rebar.loop)/V(simple.diff))')+
    facet_wrap(~Variable,scales='free_x')

