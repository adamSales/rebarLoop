### put predictions into main dataset
df <- read.csv('data/updated_exp_predictions.csv')
pred <- read.csv('Python Code/Study_1/model_predictions.csv')

pred <- subset(pred,pred$next_is_experiment == 1)
pred$link <- as.character(pred$link)
df$id <- as.character(paste(df$User.ID,df$problem_set,sep='_'))

dat <- df
dat$p_complete <- rep(NA,nrow(dat))

for (i in 1:nrow(df)) {
  #i = 1
  p = subset(pred,as.character(pred$link) == df$id[i])
  if (nrow(p) > 1) {
   print(paste('problem with',df$id,sep=' '))
  }
  else if (nrow(p) == 1) {
    dat$p_complete[i] <- p$p_complete[1]
    dat$p_inverted_mastery[i] <- p$p_inv_mastery[1]
  }
  else {
    sub_p <- subset(pred,pred$experiment==df$problem_set[i])
    dat$p_complete[i] <- mean(na.omit(sub_p$complete))
    dat$p_inverted_mastery[i] <- mean(na.omit(sub_p$p_inv_mastery))
  }
}

dev.off()
dat$p_complete <- ifelse(dat$p_complete > 1, 1,dat$p_complete)
dat$p_complete <- ifelse(dat$p_complete < 0, 0,dat$p_complete)
hist(dat$p_complete)
dat$p_complete <- (dat$p_complete)

write.csv(dat,'updated_exp_predictions.csv',row.names=FALSE)

