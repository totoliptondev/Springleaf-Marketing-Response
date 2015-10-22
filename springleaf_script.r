library(readr)
library(xgboost)
set.seed(1)

setwd("C:/Users/tao/Desktop/Springleaf Marketing Response")

cat("reading the train and test data\n")
train <- read_csv("train.csv/train.csv")
test  <- read_csv("test.csv/test.csv"  )

feature.names <- names(train)[2:ncol(train)-1]

cat("sampling train to 4GB memory limitations\n")
train <- train[sample(nrow(train), 100000),]

# cat("replacing missing values with -1\n")
train[is.na(train)] <- -1

cat("assuming text variables are categorical & replacing them with numeric ids\n")

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

gc()


cat("training a XGBoost classifier\n")

param <- list("objective" = "binary:logistic",    
              "eval_metric" = "auc",    
              "max_depth" = 17,   
              "eta" = 0.01,             # step size shrinkage 
              "subsample" = 0.8,        # part of data instances to grow tree 
              "colsample_bytree" = 0.7  # subsample ratio of columns when constructing each tree 
             )

clf <- xgboost(data        = data.matrix(train[ ,feature.names]),
               label       = train$target,
               param       = param,
               nrounds     = 500,
               early.stop.round = 15)

cat("making predictions in batches due to 4GB memory limitation\n")

submission <- data.frame(ID=test$ID)
submission$target <- NA 

for (rows in split(1:nrow(test), ceiling((1:nrow(test))/1000))) {
  # setting NA in tmp to -1 in batch due to memory limitation of creating is.na(test).
  tmp = data.matrix(test[rows,feature.names])
  tmp[is.na(tmp)] <- -1
  submission[rows, "target"] <- predict(clf, tmp)
}

cat("saving the submission file\n")
write_csv(submission, "xgboost_submission.csv")