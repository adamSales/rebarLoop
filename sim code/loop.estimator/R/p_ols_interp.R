#' OLS Imputation of Potential Outcomes in Paired Experiments
#'
#' This function is used to impute potential outcomes for the \code{p_loop} function. 
#' Leaves out each pair and imputes its potential outcomes using ordinary least squares
#' regression on the remaining pairs. This function interpolates between the \code{p_ols_ab} and \code{p_ols_s} methods.
#' @param assigned A matrix of pair experimental data that his been processed by the \code{pair} function.
#' @export

p_ols_interp = function(assigned){
  pred1 = p_ols_ab(assigned)
  pred2 = p_ols_s(assigned)
  Tr = assigned$Tr
  a = assigned$Y1[Tr == 1]-assigned$Y2[Tr == 1]
  b = assigned$Y2[Tr == 0]-assigned$Y1[Tr == 0]
  ab = p_interp(Tr,a,b,pred1,pred2)[[1]]
  return(ab)
}

