library(dplyr)
library(tidyr)
library(readr)
library(SuperLearner)

train <- read_csv('rebar_ft_training.csv')
train$totTime <- train$last_activity-train$start_time
train <- filter(train,totTime>60)


test <- read_csv('rebar_ft_test.csv')
test$totTime <- test$last_activity-test$start_time
test <- filter(test,totTime>60|is_experiment==1)


dat <- read.csv('../../../updated_exp_predictions.csv')

dat <- read_csv('g:/My Drive/assistments/ASSISTments_dataset_22_experiments.csv')

test <- test%>%group_by(user_id)%>%mutate(seq=n():1)

dat2 <- subset(dat,(User.ID%in%test$user_id)&(problem_set%in%test$sequence_id))

testS <- split(test,test$user_id)

test2 <- list()
#for(i in 1:nrow(dat2)) test2[[i]] <-
test2 <- lapply(1:nrow(dat2),function(i)
    filter(testS[[as.character(dat2$User.ID[i])]], seq> seq[which(sequence_id==dat$problem_set[i])[1]])%>%
        mutate(ps_ex=dat2$problem_set[i],nprob=n()))

npast <- sapply(test2,nrow)

test2 <- test2[npast>=4]

test2 <- do.call('rbind',test2)

test2$id <- paste0(test2$user_id,'_',test2$ps_ex)

test2 <- test2%>%group_by(id)%>%mutate(seq=n():1)%>%ungroup()%>%filter(seq<=4)

test <- test2

print(sum(test2$sequence_id==test2$ps_ex))

train <- train%>%group_by(user_id)%>%mutate(seq=n():1,totProbs=n())

aaa <- train%>%group_by(user_id)%>%summarize(n=totProbs[1])


train <- train%>%filter(totProbs>3,seq<5)

train2 <- train%>%group_by(user_id)%>%summarize(Y=complete[n()])
#trainS <- split(train,train$user_id)

for(s in 2:4){
    train3 <- train%>%group_by(user_id)%>%filter(seq==s)%>%
        dplyr::select(complete,problems_started,problems_completed,inverse_mastery_speed,
               percent_correct, starts_with('avg_'))
    names(train3)[-1] <- paste0(names(train3)[-1],s)
    train2 <- full_join(train2,train3)
}

weird <- function(x) sum(x/sum(!is.na(x)),na.rm=TRUE)

for(i in 1:ncol(train2)) train2[is.na(train2[,i]),i] <- weird(train2[,i])

test2 <- test%>%group_by(id)%>%summarize(user_id=user_id[1],ps_ex=ps_ex[1])

test2 <- test%>%group_by(id)%>%filter(seq==1)%>%
  dplyr::select(user_id,ps_ex,complete,problems_started,problems_completed,
    inverse_mastery_speed,percent_correct,starts_with('avg_'))
names(test2)[-1] <- paste0(names(test2)[-1],2)

for(s in 1:3){
    test3 <- test%>%group_by(id)%>%filter(seq==s)%>%
      dplyr::select(complete,problems_started,problems_completed,
        inverse_mastery_speed,percent_correct,starts_with('avg_'))
    names(test3)[-1] <- paste0(names(test3)[-1],s+1)
    test2 <- full_join(test2,test3)
}


for(i in 1:ncol(test2)) if(any(is.na(test2[,i]))) test2[is.na(test2[,i]),i] <- weird(test2[,i])

m1 <- glm(Y~.-user_id,family=binomial,data=train2)



model <- SuperLearner(train2$Y,as.data.frame(train2[,-c(1:2)]),
  newX=as.data.frame(test2[,-1]),
  family=binomial,
  SL.library=c('SL.bayesglm','SL.gam','SL.gbm',
    'SL.glmnet','SL.nnet','SL.randomForest',
    'SL.stepAIC','SL.glm'))
save(model,file='try.RData')


