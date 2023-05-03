library(tidyverse)
library(randomForest)

rem <- read_csv('Python Code/Study_1/resources/remnant.csv')

### is this the data used in the model?
# ========================================
## ======  Label Description  =============
## ========================================
## -- Number of Samples: 130678
## -- Number of Label Sets: 2
## ---- 1: 1 Label
## ---- 2: 1 Label
## ========================================

n_distinct(rem$user_id)

### looks right

### what are the covariates and labels, etc.?
## from Study_1/Study1_Model.ipynb
## key = [1]
names(rem)[2] ## python starts counting at 0
## label = [[13], [15], [12]]
names(rem)[c(14,16,13)]
## cov = [7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
names(rem)[c(7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)+1]
## iden = [0, 2, 3, 6]
## sortby = [4]


remStud <- rem[,c(1,7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)+1]%>%
  group_by(user_id)%>%
  summarize(across(everything(),mean,na.rm=TRUE),nRec=n())

remStud <- mutate(remStud,across(everything(),~ifelse(is.na(.),mean(.,na.rm=TRUE),.)))

save(remStud,file='results/remStudAvgStudy1.RData')
## plan: stack remnant against each experiment, 1 by 1
## fit RF predicting remnant vs experiment
## look at misclassification rate for experiment ---> low misclassification rate means poor overlap

ex <- read_csv('Python Code/Study_1/resources/experimental.csv')

dim(ex)

all(names(rem)%in%names(ex))

exStud <- ex[,c('experiment',names(rem)[c(1,7, 8, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)+1])]%>%
  group_by(user_id,experiment)%>%
  summarize(across(everything(),mean,na.rm=TRUE),nRec=n())

### mean imputation
exStud <- exStud%>%
  group_by(experiment)%>%
  mutate(across(everything(),~ifelse(is.na(.),mean(.,na.rm=TRUE),.)))%>%
  ungroup()


save(exStud,file='results/exStudAvgStudy1.RData')

exStud$ex <- TRUE
remStud$ex <- FALSE


rfMods <- map(unique(exStud$experiment),
              ~exStud%>%
                filter(experiment==.x)%>%
                select(-user_id,-experiment)%>%
                bind_rows(select(remStud,-user_id))%>%
                mutate(ex=factor(ex))%>%
                randomForest(ex~.,data=.))

save(rfMods,'results/overlapModsStudy1.RData')
