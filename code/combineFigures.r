

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(scales)
library(tikzDevice)

subwayPalette <- c('#e6194b','#3cb44b','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#aaffc3','#808000','#ffd8b1','#000080','#808080','#ffe119')



## load('newResults.RData')

## fullres1 <- list()
## for(i in 1:nrow(res)){
##     rrr <- res[i,]
##     r <- cbind(est=rrr[c('regEst1','rebarEst1','rebarLoopOLS1','predCovs1','superReLoop1','justCovs1')],
##                se=sqrt(rrr[c('regEst2','rebarEst2','rebarLoopOLS2','predCovs2','superReLoop2','justCovs2')]))
##     r <- cbind(r,improvement=1-r[,'se']/r[1,'se'])
##     rownames(r) <- c('simpDiff','rebar','strat1','strat2','strat3','justCovs')
##     fullres1[[as.character(rrr[1])]] <- r
## }

## load('../../results/fullres.RData')
## fullres0 <- fullres

## fullres <- c(fullres0,fullres1)

## rnk <- c(LETTERS,letters)[rank(map_dbl(fullres,~.['simpDiff','se']))]

## names(fullres) <- rnk

## pd <- do.call('rbind',
##   lapply(names(fullres),
##     function(x) cbind(as.data.frame(fullres[[x]]),
##       method=factor(rownames(fullres[[x]]),
##         levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
##       rnk=x)))


pd <- pd%>%
 select(-improvement)%>%
 rename(experiment=rnk)%>%
 filter(method!='strat2')%>%
 mutate(method=fct_recode(.$method,`ReLoop-OLS`='strat1',`ReLoop-EN`='strat3',Loop='justCovs',Rebar='rebar'))



#pd = read_csv( "plottingData.csv" )
pd$method = factor( pd$method, c( "ReLoop-EN", "ReLoop-OLS", "Loop", "Rebar", "simpDiff" ), ordered = TRUE )

pd3 <- pd%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='simpDiff']^2/se^2,
         percent = se / se[method=='simpDiff'] ) %>%
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
methodOrd=rev(c('Simple Difference','Rebar','Loop','ReLoop-OLS','ReLoop-EN'))
comparisons$method1 <- factor(comparisons$method1,levels=methodOrd)
comparisons$method2 <- factor(comparisons$method2,levels=methodOrd)

compLevs=rev(unique(comparisons$comp[order(as.numeric(comparisons$method2),as.numeric(comparisons$method1))]))

comparisons$comp <- factor(comparisons$comp,levels=compLevs)



#### sd vs rebar, relOOP*, rebar vs relOOP*
tikz('fig4combined.tex',width=6.4,height=2,standAlone=T)

p <- comparisons%>%
    filter(method1%in%c('ReLoop-OLS','Rebar'),method2%in%c('ReLoop-EN','Rebar','Simple Difference'))%>%
    ggplot(aes(ssMult))+#,fill=exGroup))+
    geom_dotplot( method="histodot", binwidth = .047 )  +
    labs( x = "Relative Ratio of Sample Variances", y="" ) +
    geom_vline( xintercept = 1, col="red" ) +
    facet_wrap(~comp,nrow=1)+
    theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank())


print(p)

dev.off()



tikz('fig5combined.tex',width=6.4,height=2,standAlone=TRUE)

p <- comparisons%>%
  filter(method1%in%c('ReLoop-EN','Loop'),method2%in%c('Loop','Simple Difference'))%>%
     ggplot(aes(ssMult))+#,fill=exGroup))+
    geom_dotplot( method="histodot", binwidth = .047 )  +
    labs( x = "Relative Ratio of Sample Variances", y="" ) +
    geom_vline( xintercept = 1, col="red" ) +
    facet_wrap(~comp,nrow=1)+
    theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank())
print(p)

dev.off()

#setwd('figure')
try(system('pdflatex fig4combined.tex'))
try(system('pdflatex fig5combined.tex'))
# setwd('..')

levels(pd3$experiment) <- sort(unique(pd3$experiment))
ggplot(pd3,aes(experiment,se,color=method))+geom_point(position=position_dodge(width=.3),size=2)+
  scale_color_manual(values=subwayPalette)+
  scale_y_continuous(trans='log')#,breaks=c(0.005,0.01,seq(.02,.10,.02)))
ggsave('figure/seFigCombined.jpg')
