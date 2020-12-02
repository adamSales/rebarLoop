#rnk <- rank(sapply(fullres,function(x) x['strat3','improvement']))
source('code/multiplot.r')

library(gridExtra)
library(grid)

pd <- do.call('rbind',
  lapply(names(fullres),
    function(x) cbind(as.data.frame(fullres[[x]]),
      method=factor(rownames(fullres[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x)))


## luke <- pd%>%
##   select(-improvement)%>%
##   rename(experiment=rnk)%>%
##   filter(method!='strat2')%>%
##   mutate(method=fct_recode(.$method,ReLoopStar='strat1',ReLoop='strat3',loop='justCovs'))

## write.csv(luke,'plottingData.csv',row.names=FALSE)

pd1 <- pd%>%filter(method%in%c('simpDiff','rebar','strat1'))%>%
    mutate(method=fct_recode(method,`Simple Difference`='simpDiff', `Rebar`='rebar',`ReLOOP*`='strat1'))
pd1$rnk <- factor(as.character(pd1$rnk))


ggplot(pd1,aes(rnk,se*100,color=method,group=method))+
  geom_point()+
  geom_line()+
  theme(legend.position='top',text=element_text(size=12))+
      labs(x='RCT',y='Standard Error (Percentage Points)',color=NULL,group=NULL)
ggsave('ses1.pdf',width=6.5*.95,height=4)


pd2 <- pd%>%filter(method%in%c('simpDiff','justCovs','strat3'))%>%
    mutate(method=fct_recode(method,`Simple Difference`='simpDiff', `LOOP (Within Sample)`='justCovs',`ReLOOP`='strat3'))
pd2$rnk <- factor(as.character(pd2$rnk))


ggplot(pd2,aes(rnk,se*100,color=method,group=method))+
  geom_point()+
  geom_line()+
  theme(legend.position='top',text=element_text(size=12))+
      labs(x='RCT',y='Standard Error (Percentage Points)',color=NULL,group=NULL)
ggsave('ses2.pdf',width=6.5*0.95,height=4)


is_outlier <- function(x) {
  return(x < quantile(x, 0.25) - 1.5 * IQR(x) | x > quantile(x, 0.75) + 1.5 * IQR(x))
}

pd3 <- pd%>%group_by(rnk)%>%mutate(ssMult=se[method=='simpDiff']^2/se^2)%>%ungroup()%>%
    group_by(method)%>%
        mutate(outlier = is_outlier(ssMult),
               lab=ifelse(outlier, as.character(rnk), as.character(NA))) %>%
            ungroup()%>%
    filter(method%in%c('rebar','justCovs','strat1','strat3'))%>%
        mutate(method=fct_recode(method,'LOOP (Within Sample)'='justCovs','Rebar'='rebar',
                   'ReLOOP'='strat3','ReLOOP*'='strat1'),
               method=fct_relevel(method,'LOOP (Within Sample)','Rebar','ReLOOP*','ReLOOP'))

#### boxplot
ggplot(pd3,aes(method,ssMult))+
    geom_boxplot()+
    geom_text(aes(label=lab,hjust=-.3))+#,position=position_jitter(width=.2,height=0))+
        geom_point(data=filter(pd3,!outlier),
                   aes(method,ssMult),
               position=position_jitter(width=0.2))+
    #geom_text_repel(aes(label=rnk))+
    geom_hline(yintercept=1,linetype='dotted',size=1.5)+
    labs(x=NULL,y='Equivalent Increase\n in Sample Size')+
    scale_y_continuous(labels=percent)
ggsave('boxplots.pdf',width=6*0.95,height=3)

### boxplots w overlay for talk
pd3.0 <- pd3%>%
    mutate(method=fct_relevel(method,'Rebar','ReLOOP*','ReLOOP','LOOP (Within Sample)'))
for(i in 1:4){
    pd3.1 <- pd3.0%>%mutate(ssMult2=ifelse(method%in%levels(method)[1:i],ssMult,NA))
    ggplot(pd3.1, aes(method,ssMult2))+
        geom_boxplot()+
            geom_text(aes(label=lab,hjust=-.3))+#,position=position_jitter(width=.2,height=0))+
                geom_point(data=filter(pd3.1,!outlier),
                           aes(method,ssMult2),
                           position=position_jitter(width=0.2))+
    #geom_text_repel(aes(label=rnk))+
                    geom_hline(yintercept=1,linetype='dotted',size=1.5)+
                        labs(x=NULL,y='Equivalent Increase\n in Sample Size')+
                            scale_y_continuous(breaks=seq(.5,2,.25),labels=percent)
    ggsave(paste0('boxplots',i,'.pdf'),width=6*0.95,height=3)
}


pd3.1 <- pd3.0%>%mutate(ssMult2=ifelse(method=='Rebar',ssMult,NA))
ggplot(pd3.1, aes(method,ssMult2))+
    geom_boxplot()+
    geom_text(aes(label=lab,hjust=-.3))+#,position=position_jitter(width=.2,height=0))+
        geom_point(data=filter(pd3.1,!outlier),
                   aes(method,ssMult2),
               position=position_jitter(width=0.2))+
    #geom_text_repel(aes(label=rnk))+
    geom_hline(yintercept=1,linetype='dotted',size=1.5)+
    labs(x=NULL,y='Equivalent Increase\n in Sample Size')+
    scale_y_continuous(labels=percent)
ggsave('boxplots1.pdf',width=6*0.95,height=3)



pd3 <- pd%>%group_by(rnk)%>%mutate(ssMult=se[method=='simpDiff']^2/se^2)%>%ungroup()%>%
    filter(method=='strat3')%>%arrange(ssMult)%>%mutate(x=1:n())

plot(pd3$ssMult,ylim=c(0,3),type='n')
text(pd3$ssMult,labels=pd3$rnk,ylim=c(0,3),type='b')

ggplot(pd3,aes(x,ssMult,label=rnk))+geom_text(nudge_y=0.05)+geom_line()+geom_hline(yintercept=1,linetype='dotted')

ggplot(ungroup(pd3),aes(x=as.numeric(rnk),y=ssMult))+#,color=method,group=method))+
    geom_point()+
        geom_line()+
            theme(text=element_text(size=12))+
                labs(x='RCT',y='Equivalent Increase\n in Sample Size')+#,color=NULL,group=NULL)+
                    geom_hline(yintercept=1,linetype='dotted')+
                        scale_y_continuous(labels=percent)+
                            scale_x_continuous(breaks=1:22,labels=LETTERS[1:22])

ggsave('ssMult.pdf',width=6.5*0.95,height=3)




ggplot(filter(pd,method%in%c('simpDiff','strat3')), aes(rnk,est,color=method))+
    geom_point(position=position_dodge(.4))+
        geom_errorbar(aes(ymin=est-2*se,ymax=est+2*se),position=position_dodge(.4),width=0)+
            geom_hline(yintercept=0,linetype='dotted')+
                theme(legend.position='top')+
                    labs(color=NULL,x=NULL,y='Treatment Effect')+
                        scale_color_discrete(labels=c('Simple Difference','ReLOOP'))
ggsave('estimates.pdf',width=6.5*0.95,height=4)



### "45 degree" plots

## simple difference, rebar, reloop*
pdFig4 <- pd%>%
    filter(method%in%c('simpDiff','rebar','strat1'))%>%
        select(-est,-improvement)## %>%
        ## mutate(method=fct_recode(method,'LOOP (Within Sample)'='justCovs','Rebar'='rebar',
        ##            'ReLOOP'='strat3','ReLOOP*'='strat1'),
        ##        method=fct_relevel(method,'LOOP (Within Sample)','Rebar','ReLOOP*','ReLOOP'))
rangeFig4 <- range(pdFig4$se)

pdFig4 <- pdFig4%>%spread(method,se)

plotList <- list(
    ggplot(pdFig4,aes(simpDiff,rebar))+labs(x='Simple Difference SE',y='Rebar SE'),#,title='S. D. vs Rebar'),
    ggplot(pdFig4,aes(simpDiff,strat1))+labs(x='Simple Difference SE',y='ReLOOP* SE'),#,title='S. D. vs ReLOOP*'),
    ggplot(pdFig4,aes(rebar,strat1))+labs(x='Rebar SE',y='ReLOOP* SE'))#,title='Rebar vs ReLOOP*'))
plotList <- map(plotList,
                ~.x+geom_text(aes(label=rnk))+coord_fixed(ratio = 1, xlim = rangeFig4, ylim = rangeFig4)+
                    geom_abline(slope=1,intercept=0))


comp1 <- do.call('grid.arrange',c(plotList,nrow=1))
ggsave('compare1.pdf',comp1,width=6.5*.95,height=6.5*.95/3)

## simple difference, loop, reloop
pdFig5 <- pd%>%
    filter(method%in%c('simpDiff','justCovs','strat3'))%>%
        select(-est,-improvement)## %>%
        ## mutate(method=fct_recode(method,'LOOP (Within Sample)'='justCovs','Rebar'='rebar',
        ##            'ReLOOP'='strat3','ReLOOP*'='strat1'),
        ##        method=fct_relevel(method,'LOOP (Within Sample)','Rebar','ReLOOP*','ReLOOP'))
rangeFig5 <- range(pdFig5$se)

pdFig5 <- pdFig5%>%spread(method,se)

plotList <- list(
    ggplot(pdFig5,aes(simpDiff,justCovs))+labs(x='Simple Difference SE',y='LOOP SE'),#,title='S. D. vs Rebar'),
    ggplot(pdFig5,aes(simpDiff,strat3))+labs(x='Simple Difference SE',y='ReLOOP SE'),#,title='S. D. vs ReLOOP*'),
    ggplot(pdFig5,aes(justCovs,strat3))+labs(x='LOOP SE',y='ReLOOP SE'))#,title='Rebar vs ReLOOP*'))
plotList <- map(plotList,
                ~.x+geom_text(aes(label=rnk))+coord_fixed(ratio = 1, xlim = rangeFig5, ylim = rangeFig5)+
                    geom_abline(slope=1,intercept=0))


comp2 <- do.call('grid.arrange',c(plotList,nrow=1))
ggsave('compare2.pdf',comp2,width=6.5*.95,height=6.5*.95/3)

