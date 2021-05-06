#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(scales)
library(tikzDevice)



if(!exists("fullres")) load("results/fullres.RData")

subwayPalette <- c('#e6194b','#3cb44b','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#fffac8','#800000','#aaffc3','#808000','#ffd8b1','#000080','#808080','#ffe119')

methodName=c(
               ReLOOPEN='\\trcpen',
               ReLOOP='\\trc',
               Rebar='\\trebar',
               Loop='\\tss[\\bx,\\mathrm{RF}]',
               SimpleDifference='\\tsd'
           )

#names(fullres) <- 1:33

pd <- map_dfr(names(fullres),
    function(x) cbind(as.data.frame(fullres[[x]]),
      method=factor(rownames(fullres[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x))%>%
    select(-improvement)%>%
    rename(experiment=rnk)%>%
    filter(method!='strat2')%>%
    mutate(method=fct_recode(.$method,`ReLOOP`='strat1',`ReLOOPEN`='strat3',Loop='justCovs',Rebar='rebar',`SimpleDifference`='simpDiff'))%>%
    mutate(method=factor(method,c( "ReLOOPEN", "ReLOOP", "Loop", "Rebar", "SimpleDifference" ), ordered = TRUE )
           )


pd3 <- pd%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='SimpleDifference']^2/se^2,
         percent = se / se[method=='SimpleDifference'] ) %>%
  ungroup() %>%
  group_by(method) %>%
  mutate(lab=as.character(experiment)) %>%
  ungroup()## %>%
  ## mutate(method = fct_recode( method,
  ##       				"SimpleDifference"="simpDiff",
  ##                              "ReLOOP" = "ReLoopStar",
  ##                              "Rebar" = "rebar",
  ##                              "LOOP" = "loop" ,
  ##       				"ReLOOPEN"="ReLoop") )




# Looking at relative gain
pwide = pd3 %>% dplyr::select(method, experiment, se ) %>%
  spread( method, se )

write.csv(pwide,'results/SEscombined.csv')

make_comp = function( A, B ,pwide) {
    data.frame( comp = #paste0('$\\frac{\\textrm{V(',as.character(B),')}}{\\textrm{V(', as.character(A),')}}$'),
                    paste0('\n$\\frac{\\varhat(',methodName[as.character(B)],')}{\\varhat(',methodName[as.character(A)],')}$\n'),
    experiment=pwide$experiment,
             ssMult = pwide[[B]]^2 / pwide[[A]]^2,
             method1=as.character(A), #AS added
             method2=as.character(B), #AS added
             stringsAsFactors = FALSE )
}
#make_comp( "LOOP", "SimpleDifference" )

combos = expand.grid( A = levels( pd3$method ),
                      B = levels( pd3$method ) )
combos = filter( combos, as.numeric(A) < as.numeric(B) )
combos = mutate( combos,
                 A = as.character(A),
                 B = as.character(B) )

comparisons = pmap_df( combos, make_comp ,pwide=pwide)

### get the right order
methodOrd=rev(c('SimpleDifference','Rebar','Loop','ReLOOP','ReLOOPEN'))
comparisons$method1 <- factor(comparisons$method1,levels=methodOrd)
comparisons$method2 <- factor(comparisons$method2,levels=methodOrd)

compLevs=rev(unique(comparisons$comp[order(as.numeric(comparisons$method2),as.numeric(comparisons$method1))]))

comparisons$comp <- factor(comparisons$comp,levels=compLevs)



#### sd vs rebar, relOOP*, rebar vs relOOP*
tikz('figure/fig4.tex',width=6.4,height=2,standAlone=FALSE,
     packages=
        c(
            getOption('tikzLatexPackages'),
            '\\usepackage{amsmath,amsfonts,amsthm,amssymb,thmtools}',
            '\\usepackage{bm}',
            readLines('notation.tex')
        ))

p <- comparisons%>%
    filter(method1%in%c('ReLOOP','Rebar'),method2%in%c('ReLOOPEN','Rebar','SimpleDifference'))%>%
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
        strip.text=element_text(size=12,lineheight=0.5))


print(p)

dev.off()



tikz('figure/fig5.tex',width=6.4,height=2,standAlone=FALSE,
      packages=
        c(
            getOption('tikzLatexPackages'),
            '\\usepackage{amsmath,amsfonts,amsthm,amssymb,thmtools}',
            '\\usepackage{bm}',
            readLines('notation.tex')
        ))


p <- comparisons%>%
  filter(method1%in%c('ReLOOPEN','Loop'),method2%in%c('Loop','SimpleDifference'))%>%
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
        strip.text=element_text(size=12,lineheight=0.5))
print(p)

dev.off()


tikz('figure/fig5alt.tex',width=6.4,height=2,standAlone=FALSE,
      packages=
        c(
            getOption('tikzLatexPackages'),
            '\\usepackage{amsmath,amsfonts,amsthm,amssymb,thmtools}',
            '\\usepackage{bm}',
            readLines('notation.tex')
        ))


p <- comparisons%>%
    filter((method1%in%c('ReLOOPEN')&method2%in%c('Loop','ReLOOP','SimpleDifference')))%>%
    mutate(comp=factor(comp,levels=unique(as.character(comp))))%>%
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
        strip.text=element_text(size=12,lineheight=0.5))
print(p)

dev.off()


## setwd('figure')
## try(system('pdflatex fig4.tex'))
## try(system('pdflatex fig5.tex'))
## setwd('..')

#pd3$experiment <- factor(pd3$experiment, levels=unique(pd3$experiment)[order(as.numeric(unique(pd3$experiment)))])
pd3%>%
    filter(method%in%c('SimpleDifference','ReLOOPEN','Loop'))%>%
    mutate(experiment=as.numeric(experiment))%>%
ggplot(aes(experiment,se,color=method))+geom_point(position=position_dodge(width=.3),size=2)+
    geom_line()+
    scale_color_manual(values=subwayPalette)+
  scale_y_continuous(trans='log',breaks=c(0.01,.02, seq(.05,.20,.05)))
ggsave('figure/seFigCombined.jpg')

## for reporting results
comparisons%>%group_by(method1,method2)%>%summarize(
                                              worse=sum(ssMult<0.975),
                                              equal=sum(abs(ssMult-1)<0.025),
                                              better=sum(ssMult>1.025),
                                              best=max(ssMult),
                                              bestPS=experiment[which.max(ssMult)],
                                              best2=sort(ssMult,decreasing=TRUE)[2],
                                              best2ps=experiment[rank(ssMult)==32],
                                              worst=min(ssMult),
                                              worstPS=experiment[which.min(ssMult)])%>%
    write_csv('results/summary.csv')
