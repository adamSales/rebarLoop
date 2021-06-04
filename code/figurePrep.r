methodName=c(
               ReLOOPEN='\\trcpen',
               ReLOOP='\\trc',
               Rebar='\\trebar',
               Loop='\\tss[\\bx,\\mathrm{RF}]',
               SimpleDifference='\\tsd'
           )
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

# Looking at relative gain
pwide = pd3 %>% dplyr::select(method, experiment, se ) %>%
  spread( method, se )


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


