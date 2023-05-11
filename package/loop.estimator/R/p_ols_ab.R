#' OLS Imputation of Potential Outcomes in Paired Experiments
#'
#' This function is used to impute potential outcomes for the \code{p_loop} function. 
#' Leaves out each pair and imputes its potential outcomes using ordinary least squares
#' regression on the remaining pairs. Pairs are treated as units when making predictions.
#' @param assigned A matrix of pair experimental data that his been processed by the \code{pair} function.
#' @export

p_ols_ab = function(assigned){
  k = nrow(assigned)
  
  data_a = reorient(assigned)
  assigned_b = assigned
  nvar = (ncol(assigned)-4)/2
  
  assigned_b[,c(5:(5+nvar-1),(5+nvar):ncol(assigned_b))] = assigned_b[,c((5+nvar):ncol(assigned_b),5:(5+nvar-1))]
  data_a$W = data_a$Y1 - data_a$Y2
  a = b = rep(0,k)
  
  for(i in 1:k){
    ols_a = lm(W ~ ., data_a[-i,-c(1:3)])
    a[i] = predict(ols_a,assigned[i,])
    b[i] = predict(ols_a,assigned_b[i,])
  }
  return(cbind(a,b))
}
