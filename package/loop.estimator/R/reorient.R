#' Format Paired Data
#'
#' This function takes a data frame that has been processed by the \code{pair} function and formats it such that the treated unit comes first
#' in each row.
#' @param assigned A matrix of pair experimental data that his been processed by the \code{pair} function.
#' @param reparam If set to \code{TRUE}, covariates will be parameterized as the pairwise differences and means for each covariate.
#' @export

reorient = function(assigned,reparam = FALSE){
  out = assigned
  nvar = (ncol(out)-4)/2
  Tr = assigned$Tr
  
  if(reparam == FALSE){
    k = nrow(out)
    for(i in 1:k){
      if(out$Tr[i] == 0){
        out[i,2:3] = out[i,3:2]
        out[i,c(5:(5+nvar-1),(5+nvar):ncol(out))] = out[i,c((5+nvar):ncol(out),5:(5+nvar-1))]
      }
    }
  } else {
    out[,2] = ifelse(Tr == 1,assigned$Y1,assigned$Y2)
    out[,3] = ifelse(Tr == 1,assigned$Y2,assigned$Y1)
    out[,5:(4+nvar)] = (2*out$Tr-1)*out[,5:(4+nvar)]
  }
  out = dplyr::select(out,-Tr)
  return(out)
}

