#' Random Forest Imputation of Potential Outcomes in Paired Experiments
#'
#' This function is used to impute potential outcomes for the \code{p_loop} function. 
#' Leaves out each pair and imputes its potential outcomes using random forests
#' on the remaining pairs. Individuals are treated as units when making predictions.
#' @param assigned A matrix of pair experimental data that his been processed by the \code{pair} function
#' @export

p_rf_s = function(assigned){
  Tr = assigned$Tr
  dat = reorient(assigned)
  k = nrow(dat)
  q = (ncol(dat)-3)/2
  dat1 = data.frame(dat[,c(2,4:(4+q-1))])
  dat0 = data.frame(dat[,c(3,(4+q):ncol(dat))])
  obs1 = assigned[,c(2,5:(5+q-1))]
  obs0 = assigned[,c(3,(5+q):ncol(assigned))]
  
  varnames = c("Y",paste("V",1:q,sep = ""))
  colnames(dat1) = colnames(dat0) = varnames
  colnames(obs1) = colnames(obs0) = varnames
  
  a = b = rep(0,k)
  for(i in 1:k){
    f1 = randomForest::randomForest(Y ~ ., dat1[-i,]) # treatment forest
    f0 = randomForest::randomForest(Y ~ ., dat0[-i,]) # control forest
    
    that1 = predict(f1,obs1[i,])
    chat1 = predict(f0,obs1[i,])
    that2 = predict(f1,obs0[i,])
    chat2 = predict(f0,obs0[i,])
    
    a[i] = (that1 - chat2)
    b[i] = (that2 - chat1)
    
  }
  return(cbind(a,b))
}

