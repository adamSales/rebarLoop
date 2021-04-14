#' Variance Estimate for the P-LOOP Estimator
#'
#' This function implements the variance estimate for the P-LOOP estimator and is used within \code{p_loop}.
#' @param assigned A matrix of pair experimental data that his been processed by the \code{pair} function
#' @param a A vector of the treatment minus control differences for the pairs where the first unit is assigned to treatment.
#' @param b A vector of the treatment minus control differences for the pairs where the second unit is assigned to treatment.
#' #' @export

p_loop_var = function(assigned,a,b){
  W_a = assigned$Y1[assigned$Tr == 1] - assigned$Y2[assigned$Tr == 1]
  W_b = assigned$Y2[assigned$Tr == 0] - assigned$Y1[assigned$Tr == 0]
  S_a = sum((a[assigned$Tr == 1] - W_a)^2)
  S_b = sum((b[assigned$Tr == 0] - W_b)^2)
  N = nrow(assigned)
  varhat = (S_a + S_b)/N^2
}

