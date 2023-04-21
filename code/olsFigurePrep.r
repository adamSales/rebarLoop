methodName=c(
               ReLOOPEN='\\hat{\\tau}^{\\mathrm{SS}}[\\tilde{\\mathbf{x}},\\mathrm{EN}]',
               ReLOOP='\\hat{\\tau}^{\\mathrm{SS}}[x^r,\\mathrm{LS}]',
               Rebar='\\trebar',
               Loop='\\hat{\\tau}^{\\mathrm{SS}}[\\mathbf{x},\\mathrm{RF}]',
               SimpleDifference='\\hat{\\tau}^{\\mathrm{DM}}'
           )

methodNameLin=c(
               ReLOOPEN='\\tau^{Lin}[\\tilde{x}]',
               ReLOOP='\\tau^{Lin}[x^r]',
               Loop='\\tau^{Lin}[\\mathbf{x}]',
               SimpleDifference='\\hat{\\tau}^{\\mathrm{DM}}'
           )

methodNamePoor=c(
               ReLOOPEN='\\hat{\\beta}[\\tilde{x}]',
               ReLOOP='\\hat{\\beta}[x^r]',
               Loop='\\hat{\\beta}[\\mathbf{x}]',
               SimpleDifference='\\hat{\\tau}^{\\mathrm{DM}}'
           )


if(!exists('ols')) load('results/ols.RData')

lin <- sapply(ols,
              function(x){
                x <- x[c('simpDiff','reloopLin','reloopPlusLin','lin'),]
                rownames(x) <- c('simpDiff','strat1','strat3','justCovs')
                x},
              simplify=FALSE)

poor <- sapply(ols,
              function(x){
                x <- x[c('simpDiff','reloopPoor','reloopPlusPoor','ancova'),]
                rownames(x) <- c('simpDiff','strat1','strat3','justCovs')
                x},
              simplify=FALSE)


pdLin <- map_dfr(names(lin),
    function(x) cbind(as.data.frame(lin[[x]]),
                      method=factor(rownames(lin[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x))%>%
    select(-improvement)%>%
    rename(experiment=rnk)%>%
    filter(method!='strat2')%>%
    mutate(method=fct_recode(.$method,`ReLOOP`='strat1',`ReLOOPEN`='strat3',Loop='justCovs',Rebar='rebar',`SimpleDifference`='simpDiff'))%>%
    mutate(method=factor(method,c( "ReLOOPEN", "ReLOOP", "Loop", "Rebar", "SimpleDifference" ), ordered = TRUE )
           )

pdPoor <- map_dfr(names(poor),
    function(x) cbind(as.data.frame(poor[[x]]),
                      method=factor(rownames(poor[[x]]),
        levels=c('simpDiff','justCovs','rebar','strat1','strat2','strat3')),
      rnk=x))%>%
    select(-improvement)%>%
    rename(experiment=rnk)%>%
    filter(method!='strat2')%>%
    mutate(method=fct_recode(.$method,`ReLOOP`='strat1',`ReLOOPEN`='strat3',Loop='justCovs',Rebar='rebar',`SimpleDifference`='simpDiff'))%>%
    mutate(method=factor(method,c( "ReLOOPEN", "ReLOOP", "Loop", "Rebar", "SimpleDifference" ), ordered = TRUE )
           )


pd3Lin <- pdLin%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='SimpleDifference']^2/se^2,
         percent = se / se[method=='SimpleDifference'] ) %>%
  ungroup() %>%
  group_by(method) %>%
  mutate(lab=as.character(experiment)) %>%
  ungroup()## %>%

pd3Poor <- pdPoor%>%group_by(experiment)%>%
  mutate(ssMult=se[method=='SimpleDifference']^2/se^2,
         percent = se / se[method=='SimpleDifference'] ) %>%
  ungroup() %>%
  group_by(method) %>%
  mutate(lab=as.character(experiment)) %>%
  ungroup()## %>%



                                        # Looking at relative gain
pwideLin = pd3Lin %>% dplyr::select(method, experiment, se ) %>%
  pivot_wider('experiment',names_from='method',values_from='se')

pwidePoor = pd3Poor %>% dplyr::select(method, experiment, se ) %>%
  pivot_wider('experiment',names_from='method',values_from='se')


make_comp = function( A, B ,pwide,methodName) {
    data.frame( comp = #paste0('$\\frac{\\textrm{V(',as.character(B),')}}{\\textrm{V(', as.character(A),')}}$'),
                    paste0('\n$\\frac{\\hat{\\mathbb{V}}(',methodName[as.character(B)],')}{\\hat{\\mathbb{V}}(',methodName[as.character(A)],')}$\n'),
    experiment=pwide$experiment,
             ssMult = pwide[[B]]^2 / pwide[[A]]^2,
             method1=as.character(A), #AS added
             method2=as.character(B), #AS added
             stringsAsFactors = FALSE )
}
#make_comp( "LOOP", "SimpleDifference" )

combos = expand.grid( A = levels( droplevels(pd3Lin$method )),
                      B = levels( droplevels(pd3Lin$method ) ))
combos = filter( combos, as.numeric(A) < as.numeric(B) )
combos = mutate( combos,
                 A = as.character(A),
                 B = as.character(B) )

comparisonsLin = pmap_dfr( combos, make_comp ,pwide=pwideLin,methodName=methodNameLin)
comparisonsPoor = pmap_dfr( combos, make_comp ,pwide=pwidePoor,methodName=methodNamePoor)



### get the right order
methodOrd=rev(c('SimpleDifference','Rebar','Loop','ReLOOP','ReLOOPEN'))
comparisonsLin$method1 <- factor(comparisonsLin$method1,levels=methodOrd)
comparisonsLin$method2 <- factor(comparisonsLin$method2,levels=methodOrd)

compLevs=rev(unique(comparisonsLin$comp[order(as.numeric(comparisonsLin$method2),as.numeric(comparisonsLin$method1))]))

comparisonsLin$comp <- factor(comparisonsLin$comp,levels=compLevs)


comparisonsPoor$method1 <- factor(comparisonsPoor$method1,levels=methodOrd)
comparisonsPoor$method2 <- factor(comparisonsPoor$method2,levels=methodOrd)

compLevs=rev(unique(comparisonsPoor$comp[order(as.numeric(comparisonsPoor$method2),as.numeric(comparisonsPoor$method1))]))

comparisonsPoor$comp <- factor(comparisonsPoor$comp,levels=compLevs)

### compare to reloop
load('results/SEs.RData')
load('results/fullres.RData')

crosswalk <- sapply(fullres,function(x) attributes(x)$psid)
pwideLin$experiment=vapply(pwideLin$experiment,function(ee) names(crosswalk)[crosswalk==ee],'a')

pwideCompLin <- merge(pwide,pwideLin,by='experiment',suffixes=c('','Lin'))
comps <- data.frame(A=names(pwideLin)[-c(1,2)],B=paste0(names(pwideLin)[-c(1,2)],'Lin'))

namesReloopLin <- c(methodName,setNames(methodNameLin,paste0(names(methodNameLin),'Lin')))

comparisonsReloopLin <- pmap_dfr(comps,make_comp,pwide=pwideCompLin,methodName=namesReloopLin)




pwidePoor$experiment=vapply(pwidePoor$experiment,function(ee) names(crosswalk)[crosswalk==ee],'a')

pwideCompPoor <- merge(pwide,pwidePoor,by='experiment',suffixes=c('','Poor'))
comps <- data.frame(A=names(pwideLin)[-c(1,2)],B=paste0(names(pwideLin)[-c(1,2)],'Poor'))

namesReloopPoor <- c(methodName,setNames(methodNamePoor,paste0(names(methodNameLin),'Poor')))

comparisonsReloopPoor <- pmap_dfr(comps,make_comp,pwide=pwideCompPoor,methodName=namesReloopPoor)


if(!exists('comparisons')) load('results/reloopComparisons.RData')

newcomp <- bind_rows(
  filter(comparisons,method2=='SimpleDifference',startsWith(as.character(method1),'ReLOOP')),
  filter(comparisonsPoor,method2=='SimpleDifference',startsWith(as.character(method1),'ReLOOP')),
  filter(comparisonsReloopPoor,startsWith(as.character(method1),'ReLOOP')))


newcomp$comp <- gsub('\\\\varhat','\\\\hat{\\\\mathbb{V}}',newcomp$comp)
newcomp$comp <- gsub('\\\\trcpen','\\\\hat{\\\\tau}^{\\\\mathrm{SS}}[\\\\tilde{\\\\mathbf{x}},\\\\mathrm{EN}]',newcomp$comp)
newcomp$comp <- gsub('\\\\trc','\\\\hat{\\\\tau}^{\\\\mathrm{SS}}[x^r,\\\\mathrm{LS}]',newcomp$comp)
newcomp$comp <- gsub('\\\\tsd','\\\\hat{\\\\tau}^{\\\\mathrm{DM}}',newcomp$comp)

newcomp$comp <- factor(newcomp$comp,
                       levels=c(
                         '\n$\\frac{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{DM}})}{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{SS}}[x^r,\\mathrm{LS}])}$\n',
                         '\n$\\frac{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{DM}})}{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{SS}}[\\tilde{\\mathbf{x}},\\mathrm{EN}])}$\n',
                         '\n$\\frac{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{DM}})}{\\hat{\\mathbb{V}}(\\hat{\\beta}[x^r])}$\n',
                         '\n$\\frac{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{DM}})}{\\hat{\\mathbb{V}}(\\hat{\\beta}[\\tilde{x}])}$\n',
                         '\n$\\frac{\\hat{\\mathbb{V}}(\\hat{\\beta}[x^r])}{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{SS}}[x^r,\\mathrm{LS}])}$\n',
                         '\n$\\frac{\\hat{\\mathbb{V}}(\\hat{\\beta}[\\tilde{x}])}{\\hat{\\mathbb{V}}(\\hat{\\tau}^{\\mathrm{SS}}[\\tilde{\\mathbf{x}},\\mathrm{EN}])}$\n'))
