#' P-LOOP Estimator
#'
#' Covariate-adjusted estimator of the average treatment effect in pair randomized experiments. Given a set of experimental data (including observed outcome, treatment assignment, pair assignments, and covariates), uses a method \code{pred} to impute the potential outcomes for each observation, which are then used to estimate the average treatment effect. 
#' @param Y A vector of experimental outcomes.
#' @param Tr The treatment assignment vector.
#' @param Z A matrix or vector of covariates.
#' @param P A vector encoding the pair assignments. Observations with the same value of \code{P} will be assumed to be in the same pair.
#' @param pred The prediction algorithm used to impute potential outcomes. By default, this is \code{p_rf_interp}, which uses random forests and interpolates between methods \code{p_rf_ab} and \code{p_rf_s} that treat the pairs as units or treats the individuals as units when making predictions. Another option is \code{p_ols_interp}, which uses linear regression. User written imputation methods may be used as well.
#' @param ... Arguments to be passed to the imputation method.
#' @export
#' @examples
#' N = 50 # number of pairs
#' 
#' # parameters for the data generating process
#' b1 = 10
#' b2 = -5
#' p1 = 0.9
#' p2 = 0.5
#' E = rep(0:1,N)
#' 
#' Z = rbinom(2*N,1,ifelse(E == 0,p1,p2)) # covariates
#' tau = -10 # treatment effect
#' P = c(1:N,1:N) # pair assignments
#' 
#' # generate potential outcomes
#' c = 80 + b1*E + b2*Z + rnorm(2*N,0,2)
#' t = tau + c
#' 
#' # generate treatment assignments
#' Tr = rep(0,N*2)
#' treatments = rbinom(N,1,0.5)
#' for(i in 1:N){
#'   Tr[which(P == i)] = c(treatments[i],(1-treatments[i]))
#' }
#' 
#' Y = ifelse(Tr == 1,t,c) # observed outcomes
#' 
#' # estimate the average treatment effect for the observed data Y, Tr, and Z
#' print(p_loop(Y,Tr,Z,P,p_rf_ab)) # pairs are treated as units when imputing potential outcomes
#' print(p_loop(Y,Tr,Z,P,p_rf_s)) # individuals are treated as units when imputing potential outcomes
#' print(p_loop(Y,Tr,Z,P,p_rf_interp)) # interpolation between the two methods

p_loop = function(Y,Tr,Z,P,pred = p_rf_interp,...){
  assigned = pair(Y,Tr,Z,P)
  ab = pred(assigned,...)
  varhat = p_loop_var(assigned,ab[,1],ab[,2])
  d = 0.5*(ab[,1] - ab[,2])
  tauhat = mean(Y[Tr == 1] - Y[Tr == 0]) - mean((2*assigned$Tr-1)*d)
  return(c(tauhat,varhat))
}

