dat1 <- read.csv('data/updated_exp_predictions.csv')
dat2 <- read.csv('data/newExperiments.csv')
## outcome is 'complete'
## prediction is 'p_complete'

dat2$p_complete <- dat2$pcomplete1
dat2$problem_set <- dat2$target_sequence_id

dat1$condition <- ifelse(dat1$condition=='E',1,0)

dat1$group <- 1
dat2$group <- 2

dat <- rbind(dat1[,intersect(names(dat1),names(dat2))],dat2[,intersect(names(dat1),names(dat2))])

## only keep those subjects whom treatment could possibly affect:
#dat <- subset(dat,ExperiencedCondition)

## this is the list of experiments
dat$problem_set <- as.factor(dat$problem_set)

### analysis for "complete"; drop subjects for whom "predicted" complete is NA
dat <- droplevels(dat[!is.na(dat$p_complete),])

### sample sizes?
table(dat$problem_set,dat$condition)


dat$male <- dat$Guessed.Gender=='\"Male\"'
dat$unknownGender <- dat$Guessed.Gender=='\"Uknown\"'


dat$treatment <- dat$condition

dat$residual <- dat$complete-dat$p_complete

