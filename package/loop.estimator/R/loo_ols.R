#' Leave-One-Out Linear Regression
#'
#' This function is used within the \code{loop_ols} imputation method for the \code{loop} function.
#' Given a set of outcomes \code{Y} and predictors \code{X}, this function returns the coefficients 
#' resulting from leaving each observation out and fitting ordinary least squares regression to the remaining observations.
#' 
#' @param Y A vector of outcomes.
#' @param X A matrix of predictors.
#' @export

loo_ols = function(Y,X){
  A = solve(t(X) %*% X)
  B = t(X) %*% Y
  n = nrow(X)
  p = ncol(X)
  betas = matrix(0,n,p)
  for(i in 1:n){
    x = X[i,]
    y = Y[i]
    A.temp = A - (A %*% x %*% t(-x) %*% A)/as.numeric(1+t(-x) %*% A %*% x)
    betas[i,] = A.temp %*% (B - x * y)
  }
  return(betas)
}

