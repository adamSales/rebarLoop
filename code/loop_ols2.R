loop_ols = function(Y,Tr,Z,p = 0.5){
  N = length(Y)

  olsdat = data.frame(Y,Tr,Z)
  c_t = matrix(0,N,2)

  for(i in 1:N){
    temp1 = olsdat[-i,]
    temp2 = olsdat[i,-1]
    temp2[2,] = temp2
    temp2[,1] = c(0,1)

    ols = lm(Y ~ Z*., data = temp1)
    c_t[i,] = predict(ols,temp2)
  }

  that = c_t[,2]
  chat = c_t[,1]
  mhat = (1-p)*that + p*chat
  tauhat = mean((1/p)*(Y-mhat)*Tr-(1/(1-p))*(Y-mhat)*(1-Tr))

  n_t = length(mhat[Tr == 1])
  n_c = length(mhat[Tr == 0])
  M_t = sum((that[Tr==1]-Y[Tr==1])^2)/n_t
  M_c = sum((chat[Tr==0]-Y[Tr==0])^2)/n_c
  varhat = 1/(n_t+n_c)*((1-p)/p*M_t + p/(1-p)*M_c + 2*sqrt(M_t*M_c))

  return(c(tauhat, varhat))
}
