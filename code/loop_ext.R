loop_ext = function(Y,Tr,Z,extt=extm,extc=extm,extm=NULL,p = 0.5,dropobs=NULL,Ranger=FALSE){
  # RF with Covariates and External Predictions
  c_t.Z = rf_impute(Y,Tr,cbind(Z,as.matrix(extt),as.matrix(extc)),p,dropobs=dropobs,Ranger=Ranger)
  that.Z = c_t.Z[,2]
  chat.Z = c_t.Z[,1]

  # Data for OLS with External Predictions
  N = length(Y)
  olsdat = data.frame(Y,extc,extt)
  chat = rep(0,N)
  that = rep(0,N)

  for(i in 1:N){
    # OLS with External Predictions
    olst = lm(Y ~ ., data = olsdat[-i,-2][Tr[-i] == 1,])
    olsc = lm(Y ~ ., data = olsdat[-i,-3][Tr[-i] == 0,])
    predst = predict(olst)
    predsc = predict(olsc)

    predst.rf = that.Z[-i][Tr[-i] == 1]
    predsc.rf = chat.Z[-i][Tr[-i] == 0]

    # Interpolate between estimated potential outcomes
    alphat = sum((Y[-i][Tr[-i] == 1] - predst)*(predst.rf-predst))/sum((predst.rf-predst)^2)
    alphac = sum((Y[-i][Tr[-i] == 0] - predsc)*(predsc.rf-predsc))/sum((predsc.rf-predsc)^2)

    if(alphat < 0) {
      alphat = 0
    } else if(alphat > 1){
      alphat = 1
    }
    if(alphac < 0) {
      alphac = 0
    } else if(alphac > 1){
      alphac = 1
    }

    chat[i] = alphac*chat.Z[i] + (1-alphac)*predict(olsc,olsdat[i,])
    that[i] = alphat*that.Z[i] + (1-alphat)*predict(olst,olsdat[i,])
  }

  # Obtain an Estimate
  n_t = sum(Tr == 1)
  n_c = sum(Tr == 0)
  mhat = (1-p)*that + p*chat
  tauhat = mean((1/p)*(Y-mhat)*Tr-(1/(1-p))*(Y-mhat)*(1-Tr))

  M_t = sum((that[Tr==1]-Y[Tr==1])^2)/n_t
  M_c = sum((chat[Tr==0]-Y[Tr==0])^2)/n_c
  varhat = 1/(n_t+n_c)*((1-p)/p*M_t + p/(1-p)*M_c + 2*sqrt(M_t*M_c))

  return(c(tauhat, varhat))
}

# Imputes the potential outcomes using random forests
rf_impute = function(Y,Tr,Z,p = 0.5,dropobs = NULL,Ranger=FALSE) {
  Y = as.matrix(Y)

  if(is.null(dropobs)) {
    dropobs = ifelse(length(Y) > 30, FALSE, TRUE)
  }
  if(Ranger){
    require(ranger)
    ddd1 <- data.frame(Y=Y[Tr==1,,drop=FALSE],Z[Tr==1,,drop=FALSE])
    ddd0 <- data.frame(Y=Y[Tr==0,,drop=FALSE],Z[Tr==0,,drop=FALSE])
    forest1 = ranger(Y~.,ddd1)
    forest0 = ranger(Y~.,ddd0)
  } else{
    forest1 = randomForest::randomForest(Z[Tr==1,,drop=FALSE], Y[Tr==1,,drop=FALSE])
    forest0 = randomForest::randomForest(Z[Tr==0,,drop=FALSE], Y[Tr==0,,drop=FALSE])
  }
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
  return(cbind(chat,that))
}
