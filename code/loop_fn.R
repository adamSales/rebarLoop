#' LOOP Estimator
#'
#' Covariate-adjusted estimator of the average treatment effect in randomized experiments. Given a set of experimental data (including observed outcome, treatment assignment, and covariates), uses random forests to impute the potential outcomes for each observation, which are then used to estimate the average treatment effect. 
#' @param Y A vector of experimental outcomes.
#' @param Tr The treatment assignment vector.
#' @param Z A matrix or vector of covariates.
#' @param p The probability of being assigned to treatment. Defaults to 0.5.
#' @param dropobs When set to TRUE, lowers the bootstramp sample size in the random forest by 1 when making out-of-bag predictions. By default, this is set to TRUE if the sample size is less than or equal to 30 and FALSE otherwise.
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
#' loop.results = loop(Y, Tr, Z)
#' 
#' meandiff = mean(Y[Tr == 1]) - mean(Y[Tr == 0])
#' varhat = var(Y[Tr==1])/length(Y[Tr==1]) + var(Y[Tr==0])/length(Y[Tr==0])
#' 
#' print(c("loop",loop.results))
#' print(c("Difference in Means",meandiff,varhat))

loop = function(Y, Tr, Z, p = 0.5, dropobs = NULL) {
  Y = as.matrix(Y)
  
  if(is.null(dropobs)) {
    dropobs = ifelse(length(Y) > 30, FALSE, TRUE)
  }
    
  forest1 = randomForest::randomForest(Z[Tr==1,,drop=FALSE], Y[Tr==1,,drop=FALSE])
  forest0 = randomForest::randomForest(Z[Tr==0,,drop=FALSE], Y[Tr==0,,drop=FALSE])
  that = chat = rep(0, length(Y))
  that[Tr==0] = predict(forest1, Z[Tr==0,,drop=FALSE])
  chat[Tr==1] = predict(forest0, Z[Tr==1,,drop=FALSE])
  
  if(dropobs == FALSE) {
    that[Tr==1] = predict(forest1)
    chat[Tr==0] = predict(forest0)
  } else {
    forest1a = randomForest::randomForest(Z[Tr==1,,drop=FALSE], Y[Tr==1,,drop=FALSE], 
                            sampsize = length(Y[Tr==1])-1)
    forest0a = randomForest::randomForest(Z[Tr==0,,drop=FALSE], Y[Tr==0,,drop=FALSE], 
                            sampsize = length(Y[Tr==0])-1)
    that[Tr==1] = predict(forest1a)
    chat[Tr==0] = predict(forest0a)
  }
  mhat = (1-p)*that + p*chat
  tauhat = mean((1/p)*(Y-mhat)*Tr-(1/(1-p))*(Y-mhat)*(1-Tr))
  
  n_t = length(mhat[Tr == 1])
  n_c = length(mhat[Tr == 0])
  M_t = sum((that[Tr==1]-Y[Tr==1])^2)/n_t
  M_c = sum((chat[Tr==0]-Y[Tr==0])^2)/n_c
  varhat = 1/(n_t+n_c)*((1-p)/p*M_t + p/(1-p)*M_c + 2*sqrt(M_t*M_c))
  
  return(c(tauhat, varhat))
}
