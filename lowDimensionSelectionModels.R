cat("--- setting up helper classes and libraries ----\n")

source("setupClasses.R")
library("glmnet")
library("pracma")
library("stats")
library("leaps")

cat("--- generating linear model for n=1000, p=10, and k=3 ----\n")
cat("----- error variance = 0.05 ; data variance = 1 ----\n")
cat("----- non-null betas : b1, b2, and b4 ----\n")

p <- 10L
n <- 1000L
k <- 3L
errorVar <- 0.05
dataVar <- 1.0

dataset <- SimpleLinearModel(n = n, p = p, k = k, errorVar = errorVar, dataVar = dataVar)
dataset$generate()

cat("--- running selection models 100 times ----\n")

l <- 700L
nlambdaRidge <- p + 1
nlambdaLasso <- 100

repTimes <- 100

bestSubsetBIC <- matrix(rep(0, repTimes*p), nrow = repTimes, ncol = p)
bestSubsetTestRSS <- matrix(rep(0, repTimes*p), nrow = repTimes, ncol = p)
selectedVariables <- c()

for (val in 1:repTimes) { 
  
  # scale the X values from dataset before splitting into training and test sets
  scaledDataset <- dataset$dataset
  scaledDataset[, 2:p+1] <- scale(scaledDataset[, 2:p+1])
  sets <- SplitDataset(dataset = scaledDataset, l = l)
  sets$split()
  
  # a ridge regularization model
  ridgeFit <- glmnet(x = data.matrix(sets$trainingDataset[,-1]), y = data.matrix(sets$trainingDataset$y),
                     alpha = 0, nlambda = nlambdaRidge, standardize = FALSE)
  
  # a lasso regularization model
  lassoFit <- glmnet(x = data.matrix(sets$trainingDataset[,-1]), y = data.matrix(sets$trainingDataset$y),
                     alpha = 1, nlambda = nlambdaLasso, standardize = FALSE)
  
  # the actual number of lambdas to which the glmnet generated a prediction in lasso
  nTrueLambdaLasso <- length(lassoFit$df)
  
  # an exhaustive subset selection model
  bestSubsetFit <- regsubsets(x = data.matrix(sets$trainingDataset[,-1]), y = data.matrix(sets$trainingDataset$y), 
                              method = "exhaustive", nvmax = p)
  
  # predictions over test and training datasets
  ridgeTestPredictions <- predict(ridgeFit, data.matrix(sets$testDataset[,-1]))
  ridgeTrainingPredictions <- predict(ridgeFit, data.matrix(sets$trainingDataset[,-1]))
  lassoTestPredictions <- predict(lassoFit, data.matrix(sets$testDataset[,-1]))
  lassoTrainingPredictions <- predict(lassoFit, data.matrix(sets$trainingDataset[,-1]))

  # beta regularization factors in lasso and ridge
  sumBetasLasso <- colSums(abs(lassoFit$beta))
  sumBetasRidge <- colSums(ridgeFit$beta^2)

  # mean residual errors for ridge and lasso
  ridgeTestRSS <- colMeans((matrix(rep(sets$testDataset$y, nlambdaRidge), nrow=(n-l), ncol=nlambdaRidge) - ridgeTestPredictions)^2)
  ridgeTrainingRSS <- colMeans((matrix(rep(sets$trainingDataset$y, nlambdaRidge), nrow=l, ncol=nlambdaRidge) - ridgeTrainingPredictions)^2)
  ridgeTrainingRegularizedRSS <- colMeans((matrix(rep(sets$trainingDataset$y, nlambdaRidge), nrow=l, ncol=nlambdaRidge) - ridgeTrainingPredictions)^2) + ridgeFit$lambda*sumBetasRidge
  lassoTestRSS <- colMeans((matrix(rep(sets$testDataset$y, nTrueLambdaLasso), nrow=(n-l), ncol=nTrueLambdaLasso) - lassoTestPredictions)^2)
  lassoTrainingRSS <- colMeans((matrix(rep(sets$trainingDataset$y, nTrueLambdaLasso), nrow=l, ncol=nTrueLambdaLasso) - lassoTrainingPredictions)^2)
  lassoTrainingRegularizedRSS <- colMeans((matrix(rep(sets$trainingDataset$y, nTrueLambdaLasso), nrow=l, ncol=nTrueLambdaLasso) - lassoTrainingPredictions)^2) + lassoFit$lambda*sumBetasLasso
  
  # residual errors for best subset with BIC
  singleBestSubsetTestRSS <- rep(0, p)
  for (j in 1:p){
    coefi <- coef(bestSubsetFit, id=j)
    xvars <- names(coefi)
    pred <- data.matrix(sets$testDataset[,xvars[2:(j+1)]])%*%data.matrix(coefi[2:(j+1)]) + coefi[1]
    singleBestSubsetTestRSS[j] <- sum((sets$testDataset$y - pred)^2)
  }
  selectedVariables <- rbind(selectedVariables, which(summary(bestSubsetFit)$which[3,-1]))
  bestSubsetBIC[val,] <- summary(bestSubsetFit)$bic
  bestSubsetTestRSS[val,] <- singleBestSubsetTestRSS
  
}

# first graph : plot of ridge, lasso, and best selection as function of complexity measures
par(mfrow=c(3,1))

# graph limits for plotting legend boxes
ridgeSupYlim <- max(ridgeTestRSS, ridgeTrainingRSS, ridgeTrainingRegularizedRSS) + 1
ridgeSupXlim <- max(sumBetasRidge)
lassoSupYlim <- max(lassoTestRSS, lassoTrainingRSS, lassoTrainingRegularizedRSS) + 1
lassoSupXlim <- max(lassoFit$df)
bestSubsetSupYlim <- max(summary(bestSubsetFit)$rss, summary(bestSubsetFit)$bic, singleBestSubsetTestRSS) + 100
bestSubsetInfYlim <- min(summary(bestSubsetFit)$rss, summary(bestSubsetFit)$bic, singleBestSubsetTestRSS) + 100

# ridge model plot
plot(x = sumBetasRidge, y = ridgeTestRSS, type="b",
     xlab = "Norma euclidiana dos betas", ylab = "mean RSS", col = "red", ylim = c(0, ridgeSupYlim))
lines(x = sumBetasRidge, y = ridgeTrainingRSS, type="b", col = "blue")
lines(x = sumBetasRidge, y = ridgeTrainingRegularizedRSS, type="b", col = "black")
legend(ridgeSupXlim, ridgeSupYlim, legend=c("test", "training", "regularized training"),
       col = c("red", "blue", "black"), xjust = 1, lty = 1)
title("Ridge regression")

# lasso model plot
plot(x = lassoFit$df, y = lassoTestRSS, type="b",
     xlab = "Numero de coeficientes nao nulos", ylab = "mean RSS", col = "red", ylim = c(0, lassoSupYlim))
lines(x = lassoFit$df, y = lassoTrainingRSS, type="b", col = "blue")
lines(x = lassoFit$df, y = lassoTrainingRegularizedRSS, type="b", col = "black")
legend(lassoSupXlim, lassoSupYlim, legend=c("test", "training", "regularized training"),
       col = c("red", "blue", "black"), xjust = 1, lty = 1)
title("The Lasso")

# best selection model plot
plot(x = seq(1,10), y = singleBestSubsetTestRSS, type="b",
     xlab = "Subset size", ylab = "RSS", col = "red", ylim = c(bestSubsetInfYlim, bestSubsetSupYlim))
lines(x = seq(1,10), y = summary(bestSubsetFit)$rss, type="b", col = "blue")
lines(x = seq(1,10), y = summary(bestSubsetFit)$bic, type="b", col = "black")
legend(p, bestSubsetSupYlim, legend=c("test", "training", "BIC"),
       col = c("red", "blue", "black"), xjust = 1, lty = 1)
title("Best-subset selection")

# second graph : plot best selection RSS and BIC as function of complexity for set size >= 3; also add histogram for selected variables 
par(mfrow=c(3,1))

# BIC plot for subset size >= 3
plot(x = seq(3,10), y = colMeans(bestSubsetBIC)[3:10], type="b",
     xlab = "Subset size", ylab = "avg BIC", col = "red")
title("BIC for subset size >= 3")

# test RSS plot for subset size >= 3
plot(x = seq(3,10), y = colMeans(bestSubsetTestRSS)[3:10], type="b",
     xlab = "Subset size", ylab = "avg RSS", col = "red")
title("test RSS for subset size >= 3")

# frequency of variables for subset size == 3
barplot(tabulate(selectedVariables, nbins=10), names.arg = c("x.1", "x.2", "x.3",
                                                      "x.4", "x.5", "x.6", "x.7", "x.8", "x.9", "x.10"))
title("number of variable occurrences for subset size == 3")

# clean up variables
rm(list = c("repTimes", "val", "k", "l", "n", "nlambdaRidge", "nlambdaLasso", "p", "dataVar", "errorVar", "ridgeSupYlim", "ridgeSupXlim", "scaledDataset", "xvars"))
rm(list = c("bestSubsetInfYlim", "bestSubsetSupYlim", "coefi", "j", "lassoSupXlim", "lassoSupYlim", "nTrueLambdaLasso", "pred", "sumBetasLasso", "sumBetasRidge"))