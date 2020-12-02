

#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(scales)


pd = read_csv( "plottingData.csv" )
head( pd )

table( pd$method, pd$experiment )

pd$method = factor( pd$method, c( "ReLoop", "ReLoopStar", "loop", "rebar", "simpDiff" ), ordered = TRUE )
levels( pd$method )

is_outlier <- function(x) {
  return(x < quantile(x, 0.25) - 1.5 * IQR(x) | x > quantile(x, 0.75) + 1.5 * IQR(x))
}

pd3 <- pd%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='simpDiff']^2/se^2,
         percent = se / se[method=='simpDiff'] ) %>%
  ungroup() %>%
  group_by(method) %>%
  mutate(outlier = is_outlier(ssMult),
         lab=ifelse(outlier, as.character(experiment), as.character(NA))) %>%
  ungroup()

outliers = unique( pd3$experiment[ pd3$outlier ] )
outliers
pd3 = mutate( pd3,
              lab_full = ifelse( experiment %in% outliers, as.character(experiment), as.character(NA) ) )

head( pd3 )

#labnames = c("ReLoop*", "ReLoop", "Rebar", "Loop" )
#names( labnames ) = c( "ReLoopStar", "ReLoop", "rebar", "loop" )

pd3 = mutate( pd3, method = fct_recode( method,
                               "ReLoop*" = "ReLoopStar",
                               "Rebar" = "rebar",
                               "Loop" = "loop" ) )
table( pd3$method )


# Original boxplot
ggplot(filter(pd3, method!='simpDiff'), aes(method,ssMult)) +
  geom_boxplot() +
  geom_text(aes(label=lab,hjust=-.3))+#,position=position_jitter(width=.2,height=0))+
  geom_point(data=filter(pd3,!outlier),
             aes(method,ssMult),
             position=position_jitter(width=0.2))+
  #geom_text_repel(aes(label=rnk))+
  geom_hline(yintercept=1,linetype='dotted',size=1.5)+
  labs(x=NULL,y='Equivalent Increase\n in Sample Size')



# Looking at increase in sample size, experiment by experiment
pd4 = filter( pd3, method != "simpDiff" )

ggplot( pd4, aes(x = factor(method), y = ssMult ) ) +
  geom_hline( yintercept=1, lty=1, col="grey" ) +
  geom_dotplot( stackdir="up", binaxis="y", position="dodge", binwidth=0.01) +
  geom_text(aes(label=lab_full, vjust=-2))+#,position=position_jitter(width=.2,height=0))+
  labs(x=NULL,y='Equivalent Increase in Sample Size') +
  coord_flip()


# Pairwise comparisons
head( pd4 )
p5 = pd4 %>% dplyr::select(method, experiment, percent ) %>%
  spread( method, percent )
head( p5 )

library(ggplot2)
library( GGally )
ggpairs( p5[-c(1)] )
head( pd3 )


# Make plots with best fit lines
panel.lm <- function (x, y,  pch = par("pch"), col.lm = "red",  ...) {
  ymin <- min(y)
  ymax <- max(y)
  xmin <- min(x)
  xmax <- max(x)
  ylim <- c(min(ymin,xmin),max(ymax,xmax))
  xlim <- ylim
  points(x, y, pch = pch,ylim = ylim, xlim= xlim,...)
  abline(0, 1, col = col.lm, ...)
  ok <- is.finite(x) & is.finite(y)
  if (any(ok))
    abline(lm(y[ok]~ x[ok]),
           col = col.lm, lty=2, ...)
}

# Look at how SE estimates co-vary.
# Solid line is 45', dashed is best fit
pairs(p5[-c(1)], panel=panel.lm, asp=1)


# Looking at relative gain
pwide = pd3 %>% dplyr::select(method, experiment, se ) %>%
  spread( method, se )

head( pwide )

make_comp = function( A, B ) {
    data.frame( comp = paste( as.character(A), as.character(B), sep=" v. " ),
               experiment=pwide$experiment,
             ssMult = 100 * pwide[[A]]^2 / pwide[[B]]^2,
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
head( comparisons )

#### try to make my own dotplot
### ReLoop vs Loop
pwide%>%
    mutate(ccc=Rebar^2/ReLoop^2,cccRound=round(ccc,1))%>%
        group_by(cccRound)%>%
            mutate(cccRank=rank(ccc)-1)%>%
                ungroup()%>%
                    ggplot(aes(cccRound,cccRank,label=experiment))+
                        geom_text()




ggplot( comparisons, aes( x = percent ) ) +
  facet_wrap( ~ comp, nrow=5 ) +
  geom_vline( xintercept = 100, col="red" ) +
  geom_dotplot( method="histodot", binwidth = 1 )  +
  labs( x = "100 * Relative Ratio of SEs", y="" ) +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank())


ggplot( filter( comparisons, !grepl( "simp", comp ) ), aes( x = percent ) ) +
    geom_text(aes(85,1, label=paste0(method1,' Better')))+
  facet_wrap( ~ comp ) +
  geom_vline( xintercept = 100, col="red" ) +
  geom_dotplot( method="histodot", binwidth = 1 )  +
  labs( x = "100 * Relative Ratio of SEs", y="" ) +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank())
ggsave('comparisons.png',width=6,height=4)

#### for the JSM talk

### sd vs rebar
### sd vs reloop*
### rebar vs reloop*
### reloop* vs reloop
### reloop vs loop

comp2 <- comparisons%>%filter(comp%in%c('ReLoop v. ReLoop*',
                                      #'ReLoop v. Loop',
                                      'ReLoop* v. Rebar',
                                      'ReLoop* v. simpDiff',
                                        'Rebar v. simpDiff'))%>%
    mutate(comp=factor(comp, levels=c(
                                 'Rebar v. simpDiff',
                                 'ReLoop* v. simpDiff',
                                 'ReLoop* v. Rebar',
                                 'ReLoop v. ReLoop*')))#,
                                 #'ReLoop v. Loop')))


ggplot(comp2, aes( x = percent ) ) +
    geom_text(aes(75,.5, label=paste0(method1,' Better')))+
        xlim(60,140)+
  facet_wrap( ~ comp ,scales="free_x") +
  geom_vline( xintercept = 100, col="red" ) +
  geom_dotplot( method="histodot", binwidth = 1 )  +
  labs( x = "100 * Relative Ratio of SEs", y="" ) +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank())
ggsave('comparisons.png',width=6,height=3)




# Looking at percent improvement vs. Simple Difference
sstat = pd4 %>% group_by( method ) %>%
  summarise( percent = mean( percent ) )

ggplot( pd4) +
  facet_wrap( ~ method, ncol = 1 ) +
  geom_vline( xintercept=1, lty=1, col="grey" ) +
  geom_vline( data=sstat, aes( xintercept = percent ), col="red") +
  geom_dotplot( aes( x = percent ), stackdir="up", binwidth=0.01) +
  labs(y=NULL,x='Relative size of SE to Simple Difference')



### adam added
pwide$simpDiff50 <- map_dbl(res100,~.['simpDiff','se'])
pwide$ReLoop50 <- map_dbl(res100,~.['strat3','se'])
pwide$Loop50 <- map_dbl(res100,~.['justCovs','se'])

comparisons$n <- 'Full'


newComp <- bind_rows(
    make_comp("simpDiff50","ReLoop50"),
    make_comp("simpDiff50","Loop50"),
    make_comp("Loop50","ReLoop50"),
    make_comp("simpDiff","ReLoop"),
    make_comp("simpDiff","Loop"),
    make_comp("Loop","ReLoop"))


newComp$n <- factor(ifelse(grepl('50',newComp$comp),'50','Full'),levels=c('Full','50'))

newComp$comp <- gsub('50','',newComp$comp)
newComp$method2 <- gsub('50','',newComp$method2)

ggplot(newComp, aes( x = percent ) ) +
    geom_text(aes(125,.5, label=paste0(method2,' Better')))+
        #xlim(60,140)+
  facet_grid( n~ comp )+#,scales="free_x") +
  geom_vline( xintercept = 100, col="red" ) +
  geom_dotplot( method="histodot", binwidth = 1 )  +
  labs( x = "100 * Relative Ratio of SEs", y="" ) +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.title.y = element_blank(),
        axis.text.y= element_blank(),
        axis.ticks.y = element_blank())
ggsave('loopReLoop.png',width=6,height=3)


comparisons = pmap_df( combos, make_comp )
head( comparisons )
