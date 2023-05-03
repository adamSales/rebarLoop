library(tidyverse)

rem <- read_csv('Python Code/Study_2/resources/remnant.csv')

#### from python code
## key = [1,2,0]
## iden = [4]
## cov = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
## lab = [[3]]
## order = 7

remStud <- rem[,c(0,1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)+1]%>%
  group_by(sequence_id_target,student_user_id)%>%
  summarize(across(everything(),mean,na.rm=TRUE),nRec=n())%>%
  group_by(sequence_id_target)%>%
  mutate(across(everything(),~ifelse(is.na(.),mean(.,na.rm=TRUE),.)))

save(remStud,file='results/remStudAvgStudy2.RData')

ex <- read_csv('Python Code/Study_2/resources/experimental.csv')

all(names(rem)%in%names(ex))

exStud <-
  ex[,c(0,1,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)+1]%>%
  group_by(sequence_id_target,student_user_id)%>%
  summarize(across(everything(),mean,na.rm=TRUE),nRec=n())%>%
  group_by(sequence_id_target)%>%
  mutate(across(everything(),~ifelse(is.na(.),mean(.,na.rm=TRUE),.)))

save(exStud,file='results/exStudAvgStudy2.RData')
