## combine remnant predictions w rest of data for first 22 experiments
source('code/merge_predictions.R')

## combine remnant predictions w rest of data for remaining 11 experiments
source('code/merge2.r')


dat2$problem_set <- dat2$target_sequence_id

dat$condition <- ifelse(dat$condition=='E',1,0)

dat <- rbind(dat[,intersect(names(dat),names(dat2))],dat2[,intersect(names(dat),names(dat2))])

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

