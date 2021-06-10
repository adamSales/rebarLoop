pred2 <- read.csv('Python Code/Study_2/model_predictions.csv')
dat2 <- read.csv('data/newExperiments.csv')

dat2 <- merge(pred2,dat2,all.y=TRUE,all.x=FALSE)

dat2$p_complete <- dat2$pcomplete

