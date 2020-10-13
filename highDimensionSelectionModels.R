cat("--- setting up helper classes and libraries ----\n")

source("setupClasses.R")
library("glmnet")
library("pracma")
library("stats")
library("leaps")

cat("--- generating linear model for n=100, p=1000, and k=3 ----\n")
cat("----- error variance = 0.01 ; data variance = 1 ----\n")

p <- 1000L
n <- 100L
k <- 3L
errorVar <- 0.01
dataVar <- 1.0

dataset <- SimpleLinearModel(n = n, p = p, k = k, errorVar = errorVar, dataVar = dataVar)
dataset$generate()

cat("----- training set size = 70; ----\n")
cat("----- running selection models...; ----\n")

l <- 70L
nlambdaRidge <- 100
nlambdaLasso <- 100

# scale the X values from dataset before splitting into training and test sets
scaledDataset <- dataset$dataset
scaledDataset[, 2:p+1] <- scale(scaledDataset[, 2:p+1])
sets <- SplitDataset(dataset = scaledDataset, l = l)
sets$split()

# a ridge model
ridgeFit <- glmnet(x = data.matrix(sets$trainingDataset[,-1]), y = data.matrix(sets$trainingDataset$y),
                   alpha = 0, nlambda = nlambdaRidge, standardize = FALSE)

# a lasso model
lassoFit <- glmnet(x = data.matrix(sets$trainingDataset[,-1]), y = data.matrix(sets$trainingDataset$y),
                   alpha = 1, nlambda = nlambdaLasso, standardize = FALSE)
# the actual number of lambdas to which the glmnet generated a prediction in lasso
nTrueLambdaLasso <- length(lassoFit$df)

# a forward stepwise model
maxSizeOfSubsets <- 8
forwardStepwiseFit <- regsubsets(x = data.matrix(sets$trainingDataset[,-1]), y = data.matrix(sets$trainingDataset$y), 
                                 method = "forward", nvmax = maxSizeOfSubsets)

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

# residual errors for forward stepwise with BIC
forwardStepwiseTestRSS <- rep(0, maxSizeOfSubsets)
for (j in 1:maxSizeOfSubsets){
  coefi <- coef(forwardStepwiseFit, id=j)
  xvars <- names(coefi)
  pred <- data.matrix(sets$testDataset[,xvars[2:(j+1)]])%*%data.matrix(coefi[2:(j+1)]) + coefi[1]
  forwardStepwiseTestRSS[j] <- sum((sets$testDataset$y - pred)^2)
}

cat("--- Ridge Model Summary: \n")
print(summary(ridgeFit))
cat("--- Lasso Model Summary: \n")
print(summary(lassoFit))

# graph limits for plotting legend boxes
ridgeSupYlim = max(ridgeTestRSS, ridgeTrainingRSS, ridgeTrainingRegularizedRSS) + 1
ridgeSupXlim = max(sumBetasRidge)
lassoSupYlim = max(lassoTestRSS, lassoTrainingRSS, lassoTrainingRegularizedRSS) + 1
lassoSupXlim = max(lassoFit$df)
forwardStepwiseSupYlim = max(summary(forwardStepwiseFit)$rss, summary(forwardStepwiseFit)$bic, forwardStepwiseTestRSS) + 100
forwardStepwiseInfYlim = min(summary(forwardStepwiseFit)$rss, summary(forwardStepwiseFit)$bic, forwardStepwiseTestRSS) + 100

par(mfrow=c(3,1))

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
title("Lasso regression")

# forward stepwise model plot
plot(x = seq(1,8), y = forwardStepwiseTestRSS, type="b",
     xlab = "Tamanho do subconjunto", ylab = "RSS", col = "red", ylim = c(forwardStepwiseInfYlim, forwardStepwiseSupYlim))
lines(x = seq(1,8), y = summary(forwardStepwiseFit)$rss, type="b", col = "blue")
lines(x = seq(1,8), y = summary(forwardStepwiseFit)$bic, type="b", col = "black")
legend(p, bestSubsetSupYlim, legend=c("test", "training", "BIC"),
       col = c("red", "blue", "black"), xjust = 1, lty = 1)
title("Best subset selection")

rm(list = c("k", "l", "n", "nlambdaRidge", "nlambdaLasso", "p", "dataVar", "errorVar", "ridgeSupYlim", "ridgeSupXlim", "scaledDataset"))
rm(list = c("coefi", "j", "lassoSupXlim", "lassoSupYlim", "nTrueLambdaLasso", "pred", "sumBetasLasso", "sumBetasRidge"))