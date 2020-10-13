
# input fields:
# dataset: data frame containing y response in first column and predictions in the remaining ones
# l: number of observations to compose the training dataset. n-l observations will compose the test dataset.
# output fields:
# trainingDataset: data frame with observations randomly selected to be in training dataset.
# testDataset: data frame with observations randomly selected to be in test dataset.
SplitDataset <- setRefClass("SplitDataset",
                            fields = list(dataset = "data.frame", l = "integer",
                                          trainingDataset = "data.frame",
                                          testDataset = "data.frame"),
                            methods = list(
                              split = function() {

                                if (l >= dim(dataset)[1]) stop("l should be smaller than n")
                                n = dim(dataset)[1]

                                trainingIndexes <- sample(c(1:n), l, FALSE)
                                testIndexes <- setdiff(c(1:n), trainingIndexes)
                                
                                trainingDataset <<- dataset[trainingIndexes, ]
                                testDataset <<- dataset[testIndexes, ]

                              }
                            ))