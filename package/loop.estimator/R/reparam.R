#' Reparameterize Paired Data
#'
#' This function takes a data frame that has been processed by the \code{pair} function and formats it such that 
#' covariates are parameterized as the within pair differences and means for each covariate.
#' @param assigned A matrix of pair experimental data that his been processed by the \code{pair} function.
#' @export

# re-parameterize the covariates
reparam = function(assigned){
  nvar = (ncol(assigned)-4)/2
  k = ncol(assigned)
  Z1 = assigned[,5:(5+nvar-1)]
  Z2 = assigned[,(5+nvar):k]
  assigned[,5:(5+nvar-1)] = Z1-Z2
  assigned[,(5+nvar):k] = (Z1+Z2)/2
  names(assigned)[5:k] = c(paste("D",1:nvar,sep = ""),paste("M",1:nvar,sep = ""))
  return(assigned)
}

