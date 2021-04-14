#' Random Forest Imputation of Potential Outcomes
#'
#' This function is used to impute potential outcomes for the \code{loop} function. 
#' Fits random forests to the data and imputes the potential outcomes for each observation
#' using the out-of-bag predictions.
#' 
#' @param Y A vector of observed outcomes.
#' @param Tr The treatment assignment vector.
#' @param Z A matrix of pre-treatment covariates.
#' @param dropobs When set to TRUE, lowers the bootstrap sample size in the random forest by 1 when making out-of-bag predictions. By default, this is set to TRUE if the sample size is less than or equal to 30 and FALSE otherwise.
#' @export

loop_rf = function(Y, Tr, Z, dropobs = NULL) {
  Y = as.matrix(Y)
  
  if(is.null(dropobs)) {
    dropobs = ifelse(length(Y) > 30, FALSE, TRUE)
  }
  
  forest1 = randomForest::randomForest(Z[Tr==1,,drop=FALSE], Y[Tr==1,,drop=FALSE])
  forest0 = randomForest::randomForest(Z[Tr==0,,drop=FALSE], Y[Tr==0,,drop=FALSE])
  that = chat = rep(0, length(Y))
  that[Tr==0] = predict(forest1, Z[Tr==0,,drop=FALSE])
  chat[Tr==1] = predict(forest0, Z[Tr==1,,drop=FALSE])
  
  if(dropobs == FALSE) {
    that[Tr==1] = predict(forest1)
    chat[Tr==0] = predict(forest0)
  } else {
    forest1a = randomForest::randomForest(Z[Tr==1,,drop=FALSE], Y[Tr==1,,drop=FALSE], 
                                          sampsize = length(Y[Tr==1])-1)
    forest0a = randomForest::randomForest(Z[Tr==0,,drop=FALSE], Y[Tr==0,,drop=FALSE], 
                                          sampsize = length(Y[Tr==0])-1)
    that[Tr==1] = predict(forest1a)
    chat[Tr==0] = predict(forest0a)
  }
  return(cbind(that,chat))
}

