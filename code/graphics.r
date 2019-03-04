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



pd3 <- pd%>%group_by(rnk)%>%mutate(ssMult=se[method=='simpDiff']^2/se^2)%>%ungroup()%>%
    filter(method%in%c('rebar','justCovs','strat3'))%>%
        mutate(method=fct_recode(method,'LOOP (Within Sample)'='justCovs','Rebar'='rebar',
                   'ReLOOP'='strat3'))


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


