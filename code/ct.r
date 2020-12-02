library(readr)
library(dplyr)

m1 <- read_csv('../../CT/data/RANDstudyData/M1_algebra_rcal_20121119_fieldid.csv')
m2 <- read_csv('../../CT/data/RANDstudyData/M2_algebra_rcal_20121119_fieldid.csv')

ms <- rbind(m1[,intersect(names(m1),names(m2))],m2[,intersect(names(m1),names(m2))])

tx <- filter(ms,state=='TX')
ms%>%filter(state=='TX')%>%group_by(year,treatment)%>%summarize(nschool=n_distinct(schoolid2))

library(lme4)

library(missForest)
covs <- model.frame(~xirt+race+sex+spec_speced+spec_gifted+spec_esl+frl+schoolid2,data=tx,
  na.action=na.pass)
for(i in 1:ncol(covs)) if(is.character(covs[,i])) covs[,i] <- as.factor(covs[,i])
dat <- missForest(covs)
dat <- cbind(dat,tx[,c('y_yirt','treatment','year','classid2','scho
mmm <- lmer(y_yirt~treatment*year+poly(xirt,3)+race+sex+spec_speced+spec_gifted+spec_esl+grdlvl+frl+(1|classid2)+(1|schoolid2),data=tx)
