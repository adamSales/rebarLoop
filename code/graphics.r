#rnk <- rank(sapply(fullres,function(x) x['strat3','improvement']))

pd <- do.call('rbind',
  lapply(names(fullres),
    function(x) cbind(as.data.frame(fullres[[x]]),
      method=factor(rownames(fullres[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x)))

pd <- droplevels(subset(pd,method%in%c('simpDiff','justCovs','rebar','strat3')))
levels(pd$method) <- c('Simple Difference','LOOP (Without Remnant)','Rebar','ReLOOP')
pd$rnk <- factor(as.character(pd$rnk))
## ggplot(pd,aes(method,se,fill=method))+
##     geom_col(position='dodge')+xlab(NULL)+
##     theme(axis.title.x=element_blank(),
##         axis.text.x=element_blank(),
##           axis.ticks.x=element_blank(),
##           legend.position='top')+
##         scale_fill_manual(values=subwayPalette[1:4],name=NULL)+
##         facet_wrap(~rnk,scales="free_y")
## #ggsave('ses.pdf')

ggplot(pd,aes(rnk,se,color=method,group=method))+
  geom_point()+
  geom_line()+
  theme(legend.position='top')+
      labs(x='RCT',y='Standard Error (Percentage Points)',color=NULL,group=NULL)+
          scale_y_continuous(labels=percent)
ggsave('ses.pdf',width=6,height=4)

pd <- pd%>%group_by(rnk)%>%mutate(ssMult=se[method=='Simple Difference']^2/se^2)%>%ungroup
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


