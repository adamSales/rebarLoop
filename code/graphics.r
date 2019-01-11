#rnk <- rank(sapply(fullres,function(x) x['strat3','improvement']))

pd <- do.call('rbind',
  lapply(names(fullres),
    function(x) cbind(as.data.frame(fullres[[x]]),
      method=factor(rownames(fullres[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x)))

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



pd3 <- pd%>%group_by(rnk)%>%mutate(ssMult=se[method=='Simple Difference']^2/se^2)%>%ungroup()%>%
    filter(method%in%c('justCovs','strat3'))

ggplot(pd%>%filter(method!='Simple Difference'),aes(rnk,ssMult,color=method,group=method))+
  geom_point()+
  geom_line()+
  theme(legend.position='top')+
  labs(x='RCT',y='Sample Size Multiplier',color=NULL,group=NULL)
ggsave('ssMult.pdf')



ggplot(filter(pd,method%in%c('Simple Difference','ReLOOP')), aes(rnk,est,color=method))+
    geom_point(position=position_dodge(.4))+
        geom_errorbar(aes(ymin=est-2*se,ymax=est+2*se),position=position_dodge(.4),width=0)+
            geom_hline(yintercept=0,linetype='dotted')+
                theme(legend.position='top')+
                    labs(color=NULL,x=NULL,y='Treatment Effect')
ggsave('estimates.pdf')


