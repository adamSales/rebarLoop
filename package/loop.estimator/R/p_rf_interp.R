#' Random Forest Imputation of Potential Outcomes in Paired Experiments
#'
#' This function is used to impute potential outcomes for the \code{p_loop} function. 
#' Leaves out each pair and imputes its potential outcomes using random forests
#' on the remaining pairs. This function interpolates between the \code{p_rf_ab} and \code{p_rf_s} methods.
#' @param assigned A matrix of pair experimental data that his been processed by the \code{pair} function.
#' @param reparam If set to \code{TRUE}, covariates will be parameterized as the pairwise differences and means for each covariate when treating the pairs as units.
#' @export

p_rf_interp = function(assigned,reparam = TRUE){
  pred1 = p_rf_ab(assigned,reparam)
  pred2 = p_rf_s(assigned)
  Tr = assigned$Tr
  a = assigned$Y1[Tr == 1]-assigned$Y2[Tr == 1]
  b = assigned$Y2[Tr == 0]-assigned$Y1[Tr == 0]
  ab = p_interp(Tr,a,b,pred1,pred2)[[1]]
  return(ab)
}

