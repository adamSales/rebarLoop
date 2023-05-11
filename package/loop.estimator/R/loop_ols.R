#' OLS Imputation of Potential Outcomes
#'
#' This function is used to impute potential outcomes for the \code{loop} function. 
#' Leaves out each observation and imputes its potential outcomes using ordinary least squares
#' regression on the remaining observations.
#' 
#' @param Y A vector of observed outcomes.
#' @param Tr The treatment assignment vector.
#' @param Z A matrix of pre-treatment covariates.
#' @export

loop_ols = function(Y, Tr, Z){
  N = length(Y)
  dat = data.frame(Tr,Z)
  dat0 = data.frame(Tr = 0,Z)
  dat1 = data.frame(Tr = 1,Z)
  dm = as.matrix(model.matrix(~ . + Tr*., data = dat))
  dm0 = as.matrix(model.matrix(~ . + Tr*., data = dat0))
  dm1 = as.matrix(model.matrix(~ . + Tr*., data = dat1))
  coefs = loo_ols(Y,dm)
  
  chat = rowSums(coefs*dm0)
  that = rowSums(coefs*dm1)
  t_c = cbind(that,chat)
  
  return(t_c)
}

