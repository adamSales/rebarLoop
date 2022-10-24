
### simple difference estimator
### Neyman variance estimate
simpDiff <- function(datPS,covsPS,outcome='complete'){
  Y <- datPS[[outcome]]
  Z <- datPS$treatment
  est <- mean(Y[Z==1])- mean(Y[Z==0])
  Vhat <- var(Y[Z==1])/sum(Z==1)+var(Y[Z==0])/sum(Z==0)
  return(c(tauhat=est,varhat=Vhat))
}

### rebar estimate: simple difference on residuals
rebar <- function(datPS,covsPS)
  simpDiff(datPS,covsPS,outcome='residual')

## OLS LOOP with deep learning predictions as the only covariate
strat1 <- function(datPS,covsPS)
    with(datPS,
         loop_ols(complete,treatment,p_complete))

## RF LOOP with deep learning predictions alongside other covariates
strat2 <- function(datPS,covsPS)
    with(datPS,
         loop(complete,treatment,cbind(p_complete,covsPS)))

## Combine RF LOOP with covariates with OLS LOOP with deep learning predictions
strat3 <- function(datPS,covsPS)
    with(datPS,
         loop_ext(complete,treatment,covsPS,extm=p_complete))

## RF LOOP with covariates, no deep learning predictions
justCovs <- function(datPS,covsPS)
    with(datPS,
         loop(complete,treatment,covsPS))

## all the estimators, for a particular problem set
full <- function(ps,dat,covNames,methods=c('simpDiff','rebar','strat1','strat3','justCovs')){
    if(!'simpDiff'%in%methods) methods=c('simpDiff',methods)

    datPS <- dat[dat$problem_set==ps,]
    covsPS <- dat[dat$problem_set==ps,covNames]

    res <- sapply(methods,
                  function(FUN){
                      set.seed(613)
                      fun <- get(FUN)
                      out <- try(fun(datPS,covsPS))
                      if(inherits(out,'try-error')) return(rep(NA,2))
                      out
                  },simplify=FALSE)
    res <- do.call('rbind',res)
    res[,2] <- sqrt(res[,2])  ## standard error, not variance
    colnames(res) <- c('est','se')
    res <- cbind(res,improvement=1-res[,'se']/res['simpDiff','se'])
    attr(res,'psid') <- ps
    attr(res,'n') <- nrow(datPS)
    res
}


reloopLin <- function(datPS,covsPS){
    est <- lm_lin(complete~treatment,covariates=~p_complete,data=datPS)
    c(est=est$coefficients['treatment'],var=est$std.error['treatment']^2)
}

reloopPoor <- function(datPS,covsPS){
    est <- lm_robust(complete~treatment+p_complete,data=datPS)
    c(est=est$coefficients['treatment'],var=est$std.error['treatment']^2)
}

reloopPlusLin <- function(datPS,covsPS){
    est <- lm_lin(complete~treatment,covariates=~p_complete+as.matrix(covsPS),data=datPS)
    c(est=est$coefficients['treatment'],var=est$std.error['treatment']^2)
}

reloopPlusPoor <- function(datPS,covsPS){
    est <- lm_robust(complete~treatment+p_complete+as.matrix(covsPS),data=datPS)
    c(est=est$coefficients['treatment'],var=est$std.error['treatment']^2)
}

lin <- function(datPS,covsPS){
    est <- lm_lin(complete~treatment,covariates=~as.matrix(covsPS),data=datPS)
    c(est=est$coefficients['treatment'],var=est$std.error['treatment']^2)
}

ancova <- function(datPS,covsPS){
    est <- lm_robust(complete~treatment+as.matrix(covsPS),data=datPS)
    c(est=est$coefficients['treatment'],var=est$std.error['treatment']^2)
}




    
                                        #
#poor <- function(ps,dat,covNames,methods=('reloopLin','reloopPoor','reloopPlusLin','reloopPlusPoor','ancova')){
    
