#' Interpolates between Sets of Potential Differences in Paired Experiments
#'
#' This function is used within the \code{p_rf_interp} and \code{p_ols_interp} imputation methods for the \code{p_loop} function. 
#' Given two sets of imputed potential differences, this function returns a single set of predicted differences by picking weights
#' that minimizes the mean squared error between the observed differences and the weighted average imputed differences. This minimization
#' is done using leave-one-out cross validation.
#' 
#' @param Tr The treatment assignment vector for the pairs, where the value is 1 if the first unit in the pair is assigned to treatment and 0 otherwise.
#' @param a A vector of the treatment minus control differences for the pairs where the first unit is assigned to treatment.
#' @param b A vector of the treatment minus control differences for the pairs where the second unit is assigned to treatment.
#' @param ab1 A matrix with 2 columns. The first column contains a set of imputed differences had the first unit in each pair been treated.  The first column contains a set of imputed differences had the second unit in each pair been treated.
#' @param ab2 A second set of imputed potential differences.
#' @export

p_interp = function(Tr,a,b,ab1,ab2){
  a1 = ab1[,1]
  a2 = ab2[,1]
  b1 = ab1[,2]
  b2 = ab2[,2]
  
  nsuma = sum((a-a2[Tr == 1])*(a1[Tr == 1]-a2[Tr == 1]))
  dsuma = sum((a1[Tr == 1]-a2[Tr == 1])^2)
  nsumb = sum((b-b2[Tr == 0])*(b1[Tr == 0]-b2[Tr == 0]))
  dsumb = sum((b1[Tr == 0]-b2[Tr == 0])^2)
  
  numa = denoma = rep(0,length(a1))
  numb = denomb = rep(0,length(b1))
  numa[Tr == 1] = (a-a2[Tr == 1])*(a1[Tr == 1] - a2[Tr == 1])
  denoma[Tr == 1] = (a1[Tr == 1] - a2[Tr == 1])^2
  numb[Tr == 0] = (b-b2[Tr == 0])*(b1[Tr == 0] - b2[Tr == 0])
  denomb[Tr == 0] = (b1[Tr == 0] - b2[Tr == 0])^2
  
  alpha_a = (nsuma - numa)/(dsuma - denoma)
  alpha_b = (nsumb - numb)/(dsumb - denomb)
  alpha_ab = cbind(alpha_a,alpha_b)
  
  alpha_ab = ifelse(alpha_ab < 0,0,ifelse(alpha_ab > 1, 1, alpha_ab))
  ab = ab1*alpha_ab + ab2*(1-alpha_ab)
  out = list(ab,alpha_ab)
  return(out)
}

