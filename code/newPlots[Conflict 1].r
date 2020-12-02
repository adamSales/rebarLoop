

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(scales)
#library( GGally )


pd = read_csv( "plottingData.csv" )
pd$method = factor( pd$method, c( "ReLoop", "ReLoopStar", "loop", "rebar", "simpDiff" ), ordered = TRUE )

pd3 <- pd%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='simpDiff']^2/se^2,
         percent = se / se[method=='simpDiff'] ) %>%
  ungroup() %>%
  group_by(method) %>%
  mutate(outlier = is_outlier(ssMult),
         lab=ifelse(outlier, as.character(experiment), as.character(NA))) %>%
  ungroup()%>%
  mutate(method = fct_recode( method,
                               "ReLoop*" = "ReLoopStar",
                               "Rebar" = "rebar",
                               "Loop" = "loop" ) )





make_comp = function( A, B ) {
    data.frame( comp = paste0( 'V(',as.character(B),')/V(', as.character(A),')'),
               experiment=pwide$experiment,
             ssMult = pwide[[B]]^2 / pwide[[A]]^2,
             method1=as.character(A), #AS added
             method2=as.character(B), #AS added
              stringsAsFactors = FALSE )
}
make_comp( "Loop", "simpDiff" )

combos = expand.grid( A = levels( pd3$method ),
                      B = levels( pd3$method ) )
combos = filter( combos, as.numeric(A) < as.numeric(B) )
combos = mutate( combos,
                 A = as.character(A),
                 B = as.character(B) )

comparisons = pmap_df( combos, make_comp )

### get the right order
methodOrd=rev(c('simpDiff','Rebar','Loop','ReLoop*','ReLoop'))
comparisons$method1 <- factor(comparisons$method1,levels=methodOrd)
comparisons$method2 <- factor(comparisons$method2,levels=methodOrd)

compLevs=rev(unique(comparisons$comp[order(as.numeric(comparisons$method2),as.numeric(comparisons$method1))]))

comparisons$comp <- factor(comparisons$comp,levels=compLevs)



#### sd vs rebar, reloop*, rebar vs reloop*
comparisons%>%
    filter(method1%in%c('ReLoop*','Rebar'),method2%in%c('ReLoop*','Rebar','simpDiff'))%>%
        mutate(ssRound=round(ssMult,1))%>%
            group_by(comp,ssRound)%>%
                mutate(ssRank=rank(ssMult)-1)%>%
                    ungroup()%>%
                        ggplot(aes(ssRound,ssRank,label=experiment))+
                          geom_vline( xintercept = 1, col="red" ) +
                          geom_text()+
                          facet_wrap(~comp,nrow=1)+
                          labs(x='Sample Size Multiplier',y='')+
                          theme(legend.position = "none",
                                panel.grid = element_blank(),
                                axis.title.y = element_blank(),
                                axis.text.y= element_blank(),
                                axis.ticks.y = element_blank())+
				scale_x_continuous(breaks=seq(0.5,2,0.1),
							labels=ifelse(seq(0.5,2,0.1)%in%seq(0.5,2,0.5),seq(0.5,2,0.1),''),
							limits=c(0.5,2))
ggsave('fig4.pdf',width=6.5,height=2.5)

comparisons%>%
    filter(method1%in%c('ReLoop','Loop'),method2%in%c('Loop','simpDiff'))%>%
        mutate(ssRound=round(ssMult,1))%>%
            group_by(comp,ssRound)%>%
                mutate(ssRank=rank(ssMult)-1)%>%
                    ungroup()%>%
                        ggplot(aes(ssRound,ssRank,label=experiment))+
                            geom_vline( xintercept = 1, col="red" ) +
                            geom_text()+
                                facet_wrap(~comp,nrow=1)+
                                    labs(x='Sample Size Multiplier',y='')+
                                        theme(legend.position = "none",
                                              panel.grid = element_blank(),
                                              axis.title.y = element_blank(),
                                              axis.text.y= element_blank(),
                                              axis.ticks.y = element_blank())+
				scale_x_continuous(breaks=seq(0.5,2,0.1),
							labels=ifelse(seq(0.5,2,0.1)%in%seq(0.5,2,0.5),seq(0.5,2,0.1),''),
							limits=c(0.5,2))

ggsave('fig5.pdf',width=6.5,height=2.5)


