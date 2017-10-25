library(plyr)
library(dplyr)
library(mice)
library(xgboost)
library(caret)
library(data.table)
library(pROC)

setwd('c:/kaggle')
#retrieve train and test
train <- read.csv('train.csv', na.strings = c("", "NA"), stringsAsFactors = F)
test <- read.csv('test.csv', na.strings = c("", "NA"), stringsAsFactors = F)

#check na
which(is.na(train), arr.ind=TRUE)
which(is.na(test), arr.ind=TRUE)

#remove row that contain na
train <- train[-512001,]

#check duplicate
nrow(train) - nrow(unique(train))
nrow(test) - nrow(unique(test))


#split train
set.seed(54321)
outcome <- train$target

partition <- createDataPartition(y=outcome,
                                 p=.7,
                                 list=F)
training <- train[partition,]
testing <- train[-partition,]


#xgb matrix
withoutRV <- training %>% select(-target)
dtrain <- xgb.DMatrix(as.matrix(withoutRV),label = training$target)
withoutRV1 <- testing %>% select(-target)
dtest <- xgb.DMatrix(as.matrix(withoutRV1))


#xgboost parameters
xgb_params <- list(colsample_bytree = 0.5, #variables per tree 
                   subsample = 0.7, #data subset per tree 
                   booster = "gbtree",
                   max_depth = 7, #tree levels
                   eta = 0.1, #shrinkage
                   eval_metric = "auc", 
                   objective = "binary:logistic",
                   min_child_weight = 5,
                   gamma=0)

#cross-validation and checking iterations
set.seed(4321)
xgb_cv <- xgb.cv(xgb_params,dtrain,early_stopping_rounds = 10, nfold = 4, print_every_n = 5, nrounds=1000)


# check the model
gb_dt <- xgb.train(params = xgb_params,
                   data = dtrain,
                   verbose = 1, maximize =F,
                   nrounds = 61)

prediction <- predict(gb_dt,dtest, type = "prob")

#Check AUC (Normalized Gini Coefficient)
roc_obj <- roc(testing$target, prediction)
auc(roc_obj) *2 - 1 #0.2547793

#Apply the model
dtest1 <- xgb.DMatrix(as.matrix(test))
prediction <- predict(gb_dt,dtest1, type = "prob")

#save the file (Need to use exp and -1 to change it back)
solution <- data.frame(id = test$id, target = prediction)

#save
write.csv(solution, file = 'first_sol.csv', row.names = F)
