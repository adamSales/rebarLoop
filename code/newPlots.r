

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(scales)
library(tikzDevice)

pd = read_csv( "plottingData.csv" )
pd$method = factor( pd$method, c( "ReLoop", "ReLoopStar", "loop", "rebar", "simpDiff" ), ordered = TRUE )

pd3 <- pd%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='simpDiff']^2/se^2,
         percent = se / se[method=='simpDiff'] ) %>%
  ungroup() %>%
  group_by(method) %>%
  mutate(lab=as.character(experiment)) %>%
  ungroup()%>%
  mutate(method = fct_recode( method,
					"Simple Difference"="simpDiff",
                               "ReLOOP" = "ReLoopStar",
                               "Rebar" = "rebar",
                               "LOOP" = "loop" ,
					"ReLOOP+EN"="ReLoop") )




# Looking at relative gain
pwide = pd3 %>% dplyr::select(method, experiment, se ) %>%
  spread( method, se )


make_comp = function( A, B ,pwide) {
    data.frame( comp = paste0('$\\frac{\\text{V(',as.character(B),')}}{\\text{V(', as.character(A),')}}$'),
               experiment=pwide$experiment,
             ssMult = pwide[[B]]^2 / pwide[[A]]^2,
             method1=as.character(A), #AS added
             method2=as.character(B), #AS added
              stringsAsFactors = FALSE )
}
#make_comp( "LOOP", "Simple Difference" )

combos = expand.grid( A = levels( pd3$method ),
                      B = levels( pd3$method ) )
combos = filter( combos, as.numeric(A) < as.numeric(B) )
combos = mutate( combos,
                 A = as.character(A),
                 B = as.character(B) )

comparisons = pmap_df( combos, make_comp ,pwide=pwide)

### get the right order
methodOrd=rev(c('Simple Difference','Rebar','LOOP','ReLOOP','ReLOOP+EN'))
comparisons$method1 <- factor(comparisons$method1,levels=methodOrd)
comparisons$method2 <- factor(comparisons$method2,levels=methodOrd)

compLevs=rev(unique(comparisons$comp[order(as.numeric(comparisons$method2),as.numeric(comparisons$method1))]))

comparisons$comp <- factor(comparisons$comp,levels=compLevs)



#### sd vs rebar, relOOP*, rebar vs relOOP*
tikz('fig4.tex',width=6.4,height=2.5)
p <- comparisons%>%
    filter(method1%in%c('ReLOOP','Rebar'),method2%in%c('ReLOOP','Rebar','Simple Difference'))%>%
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
#ggsave('fig4.pdf',width=6.7,height=2.5)
print(p)
dev.off()

tikz('fig5.tex',width=6.4,height=2.5)
p <- comparisons%>%
    filter(method1%in%c('ReLOOP+EN','LOOP'),method2%in%c('LOOP','Simple Difference'))%>%
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

#ggsave('fig5.pdf',width=6.7,height=2.5)
print(p)
dev.off()

### adam added
## pwide100 <- tibble(
##   experiment=LETTERS[1:22],
##   `Simple Difference`= map_dbl(res100,~.['simpDiff','se'])[experiment],
##   `ReLOOP+` =map_dbl(res100,~.['strat3','se'])[experiment],
##   LOOP = map_dbl(res100,~.['justCovs','se'])[experiment]
## )



## newComp <- bind_rows(
##     make_comp("ReLOOP+","Simple Difference",pwide100),
##     make_comp("LOOP","Simple Difference",pwide100),
##     make_comp("ReLOOP+","LOOP",pwide100)
## )

## newComp$comp <- factor(newComp$comp,levels=compLevs)



## newComp%>%
##     filter(method1%in%c('ReLOOP+','LOOP'),method2%in%c('LOOP'))%>%#,'Simple Difference'))%>%
##         mutate(ssRound=round(ssMult,1))%>%
##             group_by(comp,ssRound)%>%
##                 mutate(ssRank=rank(ssMult)-1)%>%
##                     ungroup()%>%
##                         ggplot(aes(ssRound,ssRank,label=experiment))+
##                             geom_vline( xintercept = 1, col="red" ) +
##                             geom_text()+
##                                 facet_wrap(~comp,nrow=1)+
##                                     labs(x='Sample Size Multiplier',y='')+
##                                         theme(legend.position = "none",
##                                               panel.grid = element_blank(),
##                                               axis.title.y = element_blank(),
##                                               axis.text.y= element_blank(),
##                                               axis.ticks.y = element_blank())+
## 				scale_x_continuous(breaks=seq(0.5,2,0.1),
## 							labels=ifelse(seq(0.5,2,0.1)%in%seq(0.5,2,0.5),seq(0.5,2,0.1),''),
## 							limits=c(0.5,2))

## ggsave('fig6.pdf',width=2.23,height=2)


