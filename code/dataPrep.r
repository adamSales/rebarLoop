dat <- read.csv('updated_exp_predictions.csv')

## outcome is 'complete'
## prediction is 'p_complete'

## only keep those subjects whom treatment could possibly affect:
#dat <- subset(dat,ExperiencedCondition)

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

dat%>%group_by(problem_set)%>%summarize(corT=cor(p_complete,complete),mse=mean((complete-p_complete)^2))
cc <- dat%>%group_by(problem_set,treatment)%>%summarize(cor=cor(p_complete,complete),mse=mean((complete-p_complete)^2),r2=1-mse/var(complete),sdr=sd(complete-p_complete),sdRat=sdr/sd(complete))
cc <- cbind(
  cc%>%filter(treatment==0)%>%rename(corC=cor,mseC=mse,r2c=r2,sdC=sdr,sdRatC=sdRat),
  cc%>%filter(treatment==1)%>%select(corT=cor,mseT=mse,r2t=r2,sdT=sdr,sdRatT=sdRat))

## cor(x,y)=cov(x,y)/sd(x)sd(y)
## cov(x,y)=E(xy)-ExEy

## mse=E((x-y)^2)
## =Ex^2+Ey^2-2Exy
## =Ex^2+Ey^2-2cov(x,y)+2ExEy
## =var(x)+var(y)-2cov(x,y)+2ExEy+(Ex)^2+(Ey)^2
## =var(x)+var(y)-2cov(x,y)+(Ex+Ey)^2
## =sd(x)/sd(y)+sd(y)/sd(x)-2cor(x,y)+(Ex+Ey)^2/sd(x)sd(y)
