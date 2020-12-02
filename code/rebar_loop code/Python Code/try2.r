library(readr)
library(tidyverse)
dat <- read_csv('rebar_ft_training.csv')
datO <- dat

dat <- dat%>%filter(!is.na(start_time))%>%
    distinct(user_id,assignment_id,sequence_id,.keep_all=TRUE)%>%
    arrange(start_time)%>%mutate(time=last_activity-start_time)%>% ## total time on each assignment
    group_by(user_id)%>%mutate(nent=n())%>%filter(nent>1)%>%
        mutate(short=time<60,nshort=cumsum(short))%>%filter(!short)%>%
            mutate(nent=n())%>%filter(nent>1)


dat <- dat%>%
    mutate(rowNum=1:n(),exp=rowNum==n(),expSeq=sequence_id[nent])%>%select(-rowNum)
dat <- dat%>%filter((exp==1)|(sequence_id!=expSeq))


train <- dat%>%filter(exp==0)

train <- train%>%group_by(user_id)%>%mutate(nass=n(),totshort=max(nshort,na.rm=TRUE))%>%ungroup()
train <- train%>%group_by(user_id)%>%mutate(ims=mean(root_inverse_mastery_speed,na.rm=TRUE),pcorr=mean(percent_correct,na.rm=TRUE),avgTime=mean(time,na.rm=TRUE),
                                            bo=mean(avg_bottomout,na.rm=TRUE),at=mean( avg_attempts,na.rm=TRUE))


nrow(train)



library(lme4)
rasch <- glmer(complete~(1|user_id)+(1|sequence_id),data=train,family=binomial)
save(rasch,file='raschMod.RData')

theta <- ranef(rasch)



test1 <- dat%>%filter(exp==1)%>%select(user_id,complete)
dim(test1)
n_distinct(test1$user_id)

test1$theta <- theta$user_id[as.character(test1$user_id),'(Intercept)']

with(test1,cor.test(complete,theta))

## for comparision
mcomp <- train%>%group_by(user_id)%>%summarize(mcomp=mean(complete,na.rm=TRUE))
test1 <- left_join(test1,mcomp)
with(test1,cor.test(complete,mcomp)


for(vv in c('ims','pcorr','avgTime','bo','at')){
    train[[vv]][is.na(train[[vv]])] <- mean(train[[vv]],na.rm=TRUE)
    train[[vv]] <- scale(train[[vv]])}

raschExp <- update(rasch,.~.+ims+pcorr+avgTime+bo+at)
save(raschExp,file='raschExp.RData')


     test2 <- dat%>%filter(exp==1)%>%select(user_id,complete)
dim(test2)
n_distinct(test2$user_id)

test2$theta <- theta$user_id[as.character(test2$user_id),'(Intercept)']

with(test2,cor.test(complete,theta))

## for comparision
mcomp <- train%>%group_by(user_id)%>%summarize(mcomp=mean(complete,na.rm=TRUE))
test2 <- left_join(test2,mcomp)
with(test2,cor.test(complete,mcomp)
