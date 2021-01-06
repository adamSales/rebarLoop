#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(scales)
library(tikzDevice)

if(!exists("fullres")) load("results/fullres.RData")

subwayPalette <- c('#e6194b','#3cb44b','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#aaffc3','#808000','#ffd8b1','#000080','#808080','#ffe119')

#names(fullres) <- 1:33

pd <- map_dfr(names(fullres),
    function(x) cbind(as.data.frame(fullres[[x]]),
      method=factor(rownames(fullres[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x))%>%
    select(-improvement)%>%
    rename(experiment=rnk)%>%
    filter(method!='strat2')%>%
    mutate(method=fct_recode(.$method,`ReLOOP+OLS`='strat1',`ReLOOP+EN`='strat3',Loop='justCovs',Rebar='rebar',`Simple Difference`='simpDiff'))%>%
    mutate(method=factor(method,c( "ReLOOP+EN", "ReLOOP+OLS", "Loop", "Rebar", "Simple Difference" ), ordered = TRUE ))


pd3 <- pd%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='Simple Difference']^2/se^2,
         percent = se / se[method=='Simple Difference'] ) %>%
  ungroup() %>%
  group_by(method) %>%
  mutate(lab=as.character(experiment)) %>%
  ungroup()## %>%
  ## mutate(method = fct_recode( method,
  ##       				"Simple Difference"="simpDiff",
  ##                              "ReLOOP" = "ReLoopStar",
  ##                              "Rebar" = "rebar",
  ##                              "LOOP" = "loop" ,
  ##       				"ReLOOP+EN"="ReLoop") )




# Looking at relative gain
pwide = pd3 %>% dplyr::select(method, experiment, se ) %>%
  spread( method, se )

write.csv(pwide,'results/SEscombined.csv')

make_comp = function( A, B ,pwide) {
    data.frame( comp = paste0('$\\frac{\\textrm{V(',as.character(B),')}}{\\textrm{V(', as.character(A),')}}$'),
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
methodOrd=rev(c('Simple Difference','Rebar','Loop','ReLOOP+OLS','ReLOOP+EN'))
comparisons$method1 <- factor(comparisons$method1,levels=methodOrd)
comparisons$method2 <- factor(comparisons$method2,levels=methodOrd)

compLevs=rev(unique(comparisons$comp[order(as.numeric(comparisons$method2),as.numeric(comparisons$method1))]))

comparisons$comp <- factor(comparisons$comp,levels=compLevs)



#### sd vs rebar, relOOP*, rebar vs relOOP*
tikz('figure/fig4.tex',width=6.4,height=2,standAlone=FALSE)

p <- comparisons%>%
    filter(method1%in%c('ReLOOP+OLS','Rebar'),method2%in%c('ReLOOP+EN','Rebar','Simple Difference'))%>%
    ggplot(aes(ssMult))+#,fill=exGroup))+
    geom_dotplot( method="histodot", binwidth = .05 )  +
    labs( x = "Relative Ratio of Sample Variances", y="" ) +
    geom_vline( xintercept = 1, col="red" ) +
    facet_wrap(~comp,nrow=1)+
    theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank(),
        text=element_text(size=12),
        strip.text=element_text(size=12))


print(p)

dev.off()



tikz('figure/fig5.tex',width=6.4,height=2,standAlone=FALSE)

p <- comparisons%>%
  filter(method1%in%c('ReLOOP+EN','Loop'),method2%in%c('Loop','Simple Difference'))%>%
     ggplot(aes(ssMult))+#,fill=exGroup))+
    geom_dotplot( method="histodot", binwidth = .05 )  +
    labs( x = "Relative Ratio of Sample Variances", y="" ) +
    geom_vline( xintercept = 1, col="red" ) +
    facet_wrap(~comp,nrow=1)+
    theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank(),
        text=element_text(size=12),
        strip.text=element_text(size=12))
print(p)

dev.off()

## setwd('figure')
## try(system('pdflatex fig4.tex'))
## try(system('pdflatex fig5.tex'))
## setwd('..')

#pd3$experiment <- factor(pd3$experiment, levels=unique(pd3$experiment)[order(as.numeric(unique(pd3$experiment)))])
pd3%>%
    filter(method%in%c('Simple Difference','ReLOOP+EN','Loop'))%>%
    mutate(experiment=as.numeric(experiment))%>%
ggplot(aes(experiment,se,color=method))+geom_point(position=position_dodge(width=.3),size=2)+
    geom_line()+
    scale_color_manual(values=subwayPalette)+
  scale_y_continuous(trans='log',breaks=c(0.01,.02, seq(.05,.20,.05)))
ggsave('figure/seFigCombined.jpg')

## for reporting results
comparisons%>%group_by(method1,method2)%>%summarize(worse=sum(ssMult<0.975),equal=sum(abs(ssMult-1)<0.025),better=sum(ssMult>1.025),best=max(ssMult),worst=min(ssMult))
