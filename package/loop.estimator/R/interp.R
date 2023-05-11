#' Interpolates between Sets of Potential Outcomes
#'
#' This function is used within the \code{reloop} and \code{reloop_slow} imputation methods for the \code{loop} function. 
#' Using the input data as a training set, this function returns two weights (one for treated units and one for control units) 
#' that minimizes the mean squared error between the observed outcomes and a weighted average of two sets of potential outcomes.
#' The weight is constrained to be between 0 and 1.
#' 
#' @param Tr The treatment assignment vector.
#' @param y A vector of experimental outcomes.
#' @param tc1 A matrix with 2 columns. The first column contains a set of imputed treatment outcomes. The second column contains a set of imputed control outcomes.
#' @param tc2 A second set of imputed potential outcomes.
#' @export

interp = function(Tr,y,tc1,tc2){
  t1 = tc1[,1]
  t2 = tc2[,1]
  c1 = tc1[,2]
  c2 = tc2[,2]
  
  nsumt = sum((y[Tr == 1]-t2[Tr == 1])*(t1[Tr == 1]-t2[Tr == 1]))
  dsumt = sum((t1[Tr == 1]-t2[Tr == 1])^2)
  nsumc = sum((y[Tr == 0]-c2[Tr == 0])*(c1[Tr == 0]-c2[Tr == 0]))
  dsumc = sum((c1[Tr == 0]-c2[Tr == 0])^2)
  
  alpha_t = nsumt/dsumt
  alpha_c = nsumc/dsumc
  return(c(alpha_t,alpha_c))
}

