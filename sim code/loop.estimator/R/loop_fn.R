#' LOOP Estimator
#'
#' Covariate-adjusted estimator of the average treatment effect in randomized experiments. Given a set of experimental data (including observed outcome, treatment assignment, and covariates), uses a method \code{pred} to impute the potential outcomes for each observation, which are then used to estimate the average treatment effect. 
#' @param Y A vector of experimental outcomes.
#' @param Tr The treatment assignment vector.
#' @param Z A matrix or vector of covariates.
#' @param pred The prediction algorithm used to impute potential outcomes. By default, this is \code{loop_rf}, which uses random forests. Other options include \code{loop_ols} (which uses linear regression) and \code{reloop} (which can incorporate external predictions \code{yhat} to improve precision). As implemented, \code{reloop} is very slightly biased -- if bias is a concern, \code{reloop_slow} can be used instead. User written imputation methods may be used as well.
#' @param p The probability of being assigned to treatment. Defaults to 0.5.
#' @param ... Arguments to be passed to the imputation method. For example, \code{loop_rf} takes the argument \code{dropobs}: when set to TRUE, lowers the bootstrap sample size in the random forest by 1 when making out-of-bag predictions. By default, this is set to TRUE if the sample size is less than or equal to 30 and FALSE otherwise. For \code{reloop} and \code{reloop_slow}, external predictions \code{yhat} must be specified. This is a vector of predictions of the outcome for each participant that is obtained using an external data source.
#' @importFrom stats predict
#' @export
#' @examples
#' ## Create Simulated Data
#' N = 30 # 30 observations
#' k = 5 # 5 covariates
#' 
#' Z = matrix(runif(N*k)*10,N,k)
#' c = Z %*% runif(k) + runif(N)*5
#' t = ifelse(rowMeans(Z) < 3, c + runif(N)*5, c + runif(N)*10)
#' 
#' Tr = rbinom(N,1,0.5)
#' Y = ifelse(Tr == 0, c, t)
#' 
#' ## Run LOOP and compare with a Difference in Means
#' looprf.results = loop(Y, Tr, Z, loop_rf)
#' loopols.results = loop(Y, Tr, Z, loop_ols)
#' 
#' meandiff = mean(Y[Tr == 1]) - mean(Y[Tr == 0])
#' varhat = var(Y[Tr==1])/length(Y[Tr==1]) + var(Y[Tr==0])/length(Y[Tr==0])
#' 
#' print(c("loop rf",looprf.results))
#' print(c("loop ols",loopols.results))
#' print(c("Difference in Means",meandiff,varhat))

loop = function(Y, Tr, Z,pred = loop_rf, p = 0.5, ...) {
  Y = as.matrix(Y)
  Z = as.matrix(Z)
  
  t_c = pred(Y,Tr,Z,...)
  that = t_c[,1]
  chat = t_c[,2]
  
  mhat = (1-p)*that + p*chat
  tauhat = mean((1/p)*(Y-mhat)*Tr-(1/(1-p))*(Y-mhat)*(1-Tr))
  
  n_t = length(mhat[Tr == 1])
  n_c = length(mhat[Tr == 0])
  M_t = sum((that[Tr==1]-Y[Tr==1])^2)/n_t
  M_c = sum((chat[Tr==0]-Y[Tr==0])^2)/n_c
  varhat = 1/(n_t+n_c)*((1-p)/p*M_t + p/(1-p)*M_c + 2*sqrt(M_t*M_c))
  
  return(c(tauhat, varhat))
}
