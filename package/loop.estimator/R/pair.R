#' Format Data into Pairs
#'
#' This function takes data at the individual level and returns a data frame with one pair per row. Each row includes the pair label,
#' the observed outcomes, the treatment assignment for the first individual in the pair, and the covariates for each individual.
#' @param Y A vector of experimental outcomes.
#' @param Tr The treatment assignment vector.
#' @param Z A matrix or vector of covariates.
#' @param P A vector encoding the pair assignments. Observations with the same value of \code{P} will be assumed to be in the same pair.
#' @export

pair = function(Y,Tr,Z,P){
  nvar = ifelse(is.null(ncol(Z)),1,ncol(Z))
  data = data.frame(cbind(Y,Tr,Z,P))
  
  # Aggregate by pair
  agg = aggregate(data, by = list(P), "c")
  agg = data.frame(as.matrix(agg[,-ncol(agg)]))
  
  # Reformat
  # Switch column order
  k = ncol(agg)
  ones = seq(6,k,2)
  twos = seq(7,k,2)
  
  out = dplyr::select(agg,1,2,3,4,all_of(ones),all_of(twos))
  names(out)[1:4] = c("P","Y1","Y2","Tr")
  
  return(out)
}

