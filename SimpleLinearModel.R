
# fields:
# n: number of observations y1,y1...,yn
# p: number of predictors in matrix X x1,x2,..,xp
# k: beta parameters where bj=1 for j=1,2,k and bj=0 for j>k
SimpleLinearModel <- setRefClass("SimpleLinearModel",
                                 fields = list(n = "integer", p = "integer", k = "integer",
                                               errorVar = "numeric", dataVar = "numeric",
                                               dataset = "data.frame", beta = "matrix", epsilon = "matrix"),
                                 methods = list(
                                   generate = function() {

                                     if (k > p) stop("k should not be bigger than p")
                                     if (n < 1) stop("n should be bigger than 0")

                                     beta <<- matrix(c(rep(1, k-1), c(0,1), rep(0, (p - k - 1))), ncol = 1)

                                     # generating n*p iid samples of X ~ N(0, dataVar)
                                     X <- matrix(rnorm(n*p, mean = 0, sd = sqrt(dataVar)), nrow = n, ncol = p)

                                     # generating n iid sample of epsilon ~ N(0, errorVar)
                                     epsilon <<- matrix(rnorm(n, mean = 0, sd = sqrt(errorVar)), nrow = n, ncol = 1)

                                     # generating n observations of y = X*b + epsilon
                                     y <- X%*%beta + epsilon

                                     dataset <<- data.frame(y= y, x = X)
                                   }
                                 ))