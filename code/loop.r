loop <- function (Y, Tr, Z, p = 0.5, dropobs = NULL,reg=TRUE)
{
    Y = as.matrix(Y)
    if (is.null(dropobs)) {
        dropobs = ifelse(length(Y) > 30, FALSE, TRUE)
    }
    if(reg){
        forest1 = randomForest::randomForest(Z[Tr == 1, , drop = FALSE],
            Y[Tr == 1, , drop = FALSE])
        forest0 = randomForest::randomForest(Z[Tr == 0, , drop = FALSE],
            Y[Tr == 0, , drop = FALSE])
        typ <- 'response'

        that = chat = rep(0, length(Y))
        that[Tr == 0] = predict(forest1, Z[Tr == 0, , drop = FALSE],type=typ)
        chat[Tr == 1] = predict(forest0, Z[Tr == 1, , drop = FALSE],type=typ)
        if (dropobs == FALSE) {
            that[Tr == 1] = predict(forest1,type=typ)
            chat[Tr == 0] = predict(forest0,type=typ)
        }
        else {
            forest1a = randomForest::randomForest(Z[Tr == 1, , drop = FALSE],
                Y[Tr == 1, , drop = FALSE], sampsize = length(Y[Tr ==
                                                                    1]) - 1)
            forest0a = randomForest::randomForest(Z[Tr == 0, , drop = FALSE],
                Y[Tr == 0, , drop = FALSE], sampsize = length(Y[Tr ==
                                                                    0]) - 1)
            that[Tr == 1] = predict(forest1a)
            chat[Tr == 0] = predict(forest0a)
        }
    } else{
        forest1 = randomForest::randomForest(Z[Tr == 1, , drop = FALSE],
            as.factor(Y[Tr == 1, , drop = FALSE]))
        forest0 = randomForest::randomForest(Z[Tr == 0, , drop = FALSE],
            as.factor(Y[Tr == 0, , drop = FALSE]))
        typ <- 'prob'

        that = chat = rep(0, length(Y))
        that[Tr == 0] = predict(forest1, Z[Tr == 0, , drop = FALSE],type=typ)[,2]
        chat[Tr == 1] = predict(forest0, Z[Tr == 1, , drop = FALSE],type=typ)[,2]
        if (dropobs == FALSE) {
            that[Tr == 1] = predict(forest1,type=typ)[,2]
            chat[Tr == 0] = predict(forest0,type=typ)[,2]
        }
        else {
            forest1a = randomForest::randomForest(Z[Tr == 1, , drop = FALSE],
                as.factor(Y[Tr == 1, , drop = FALSE]), sampsize = length(Y[Tr ==
                                                                    1]) - 1)
            forest0a = randomForest::randomForest(Z[Tr == 0, , drop = FALSE],
                as.factor(Y[Tr == 0, , drop = FALSE]), sampsize = length(Y[Tr ==
                                                                    0]) - 1)
            that[Tr == 1] = predict(forest1a,type=typ)[,2]
            chat[Tr == 0] = predict(forest0a,type=typ)[,2]
        }
    }
    mhat = (1 - p) * that + p * chat
    tauhat = mean((1/p) * (Y - mhat) * Tr - (1/(1 - p)) * (Y -
        mhat) * (1 - Tr))
    n_t = length(mhat[Tr == 1])
    n_c = length(mhat[Tr == 0])
    M_t = sum((that[Tr == 1] - Y[Tr == 1])^2)/n_t
    M_c = sum((chat[Tr == 0] - Y[Tr == 0])^2)/n_c
    varhat = 1/(n_t + n_c) * ((1 - p)/p * M_t + p/(1 - p) * M_c +
        2 * sqrt(M_t * M_c))
    return(c(tauhat, varhat))
}
