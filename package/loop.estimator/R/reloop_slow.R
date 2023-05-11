#' Imputation of Potential Outcomes using External Data -- Slow Version
#'
#' This function implements the ReLOOP method for imputing potential outcomes within the \code{loop} function. 
#' This method uses both covariates \code{Z} and external predictions \code{yhat} to impute potential outcomes, 
#' which can then be used within the LOOP estimator to obtain an estimate for the average treatment effect. 
#' As implemented, \code{reloop} is very slightly biased. This function removes that bias at the cost of
#' computation time.
#' 
#' @param Y A vector of observed outcomes.
#' @param Tr The treatment assignment vector.
#' @param Z A matrix of pre-treatment covariates.
#' @param yhat A vector of external predictions.
#' @export

reloop_slow = function(Y,Tr,Z,yhat){
  N = length(Y)
  
  # Data for RF with external predictions/covariates
  rfdat = cbind(Z,as.matrix(yhat))
  
  # OLS with external predictions
  # ols t's and c's
  dat = data.frame(Tr,yhat)
  dat0 = data.frame(Tr = 0,yhat)
  dat1 = data.frame(Tr = 1,yhat)
  dm = as.matrix(model.matrix(~ . + Tr*., data = dat))
  dm0 = as.matrix(model.matrix(~ . + Tr*., data = dat0))
  dm1 = as.matrix(model.matrix(~ . + Tr*., data = dat1))
  coefs = loo_ols(Y,dm)
  
  chat = rowSums(coefs*dm0)
  that = rowSums(coefs*dm1)
  tc.OLS = cbind(that,chat)
  
  # Obtain leave-one-out predictions and interpolate
  alpha_tc = tc.RF = matrix(0,N,2)
  for(i in 1:N){
    # ols predictions
    coefs.temp = coefs[i,]
    chat.temp = dm0[-i,] %*% coefs.temp
    that.temp = dm1[-i,] %*% coefs.temp
    tc.temp = cbind(that.temp,chat.temp)
    
    # random forest predictions
    forest1 = randomForest::randomForest(rfdat[-i,][Tr[-i]==1,,drop=FALSE], as.matrix(Y[-i])[Tr[-i]==1,,drop=FALSE])
    forest0 = randomForest::randomForest(rfdat[-i,][Tr[-i]==0,,drop=FALSE], as.matrix(Y[-i])[Tr[-i]==0,,drop=FALSE])
    tc.RF[i,] = c(predict(forest1,rfdat[i,]),predict(forest0,rfdat[i,]))
    
    that.RF = chat.RF = rep(0, N-1)
    that.RF[Tr[-i]==0] = predict(forest1, rfdat[-i,][Tr[-i]==0,,drop=FALSE])
    chat.RF[Tr[-i]==1] = predict(forest0, rfdat[-i,][Tr[-i]==1,,drop=FALSE])
    that.RF[Tr[-i]==1] = predict(forest1)
    chat.RF[Tr[-i]==0] = predict(forest0)
    tc.temprf = cbind(that.RF,chat.RF)
    
    alpha_tc[i,] = interp(Tr[-i],Y[-i],tc.temprf,tc.temp)
  }
  
  alpha_tc = ifelse(alpha_tc < 0,0,ifelse(alpha_tc > 1, 1, alpha_tc))
  tc.interp = tc.RF*alpha_tc + tc.OLS*(1-alpha_tc)
  return(tc.interp)
}