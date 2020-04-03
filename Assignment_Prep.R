## Regression Trees, Bagged Trees and Random Forests

## 1. Overview

### Packages
# You will need to load the following packages:
library(sf)
library(tidyverse)
library(tmap)
library(rpart)
library(rpart.plot)
library(visNetwork)
library(caret)
library(randomForest)
library(ranger)

### Data
# You will need to load the following RData file.  
load("week17_assignment_prep.RData")

# selects variables 
oa_sf %>% st_drop_geometry() %>% 
  select(-c(code, Easting, Northing)) -> anal_data
# create training and testing data
set.seed(1234) # for reproducibility
train.index = createDataPartition(anal_data$unmplyd, p = 0.7, list = F)
# split into training and validation (testing) subsets: 
train_x = anal_data[train.index,] 
test_x = anal_data[-train.index,] 

# examine the distributions of the target variable
summary(anal_data$unmplyd[train.index])
summary(anal_data$unmplyd[-train.index])

## 2. Decision Trees: finding structure in the data 

## 3. Regression Trees 


# a regression tree can be fitted tree using the `rpart` package: 
set.seed(124)
m1 <- rpart(
  formula = unmplyd ~ .,
  data    = train_x,
  method  = "anova"
  )
m1

rpart.plot(m1, box.palette = "Or")

visTree(m1,legend=FALSE,collapse=TRUE,direction='LR')

# the error for a given alpha value
plotcp(m1)

# optimal prunings 
round(m1$cptable, 4)

# specify a minsplit of 30 and maxdepth of 12:
set.seed(123)
m2 <- rpart(
  formula = unmplyd ~ .,
  data    = train_x,
  method  = "anova",
  control = list(minsplit = 30, maxdepth = 12)
)
m2$cptable

# data frame of 189 combinations parameters:
params <- expand.grid(
  minsplit = seq(10, 30, 1),
  maxdepth = seq(4, 12, 1)
)
dim(params)
head(params)

# pass to rpart in a loop to generate a table of results:
param_test <- matrix(nrow = 0, ncol = 4)
colnames(param_test) = c("minsplit", "maxdepth", "cp", "xerror")
for (i in 1:nrow(params)) {
  # get values for row i in params
  minsplit.i <- params$minsplit[i]
  maxdepth.i <- params$maxdepth[i]
  # create the model 
  mod.i <- rpart(
    formula = unmplyd ~ .,
    data    = train_x,
    method  = "anova",
    control = list(minsplit = minsplit.i, maxdepth = maxdepth.i)
  )
  # extract the optimal complexity paramters
  min <- which.min(mod.i$cptable[, "xerror"])
  cp <- mod.i$cptable[min, "CP"] 
  # get minimum error
  xerror <- mod.i$cptable[min, "xerror"] 
  res.i = c(minsplit.i, maxdepth.i, cp, xerror)
  param_test = rbind(param_test,res.i )
  # uncomment for progress
  if (i%%10 == 0) cat(i, "\t")
}
param_test = data.frame(param_test)

# the table of results can be ordered and displayed:
head(param_test[order(param_test$xerror),])

# creates a model using these parameters 
set.seed(123)
# assign the best to a vector
vals = param_test[order(param_test$xerror)[1],]
m3 <- rpart(
  formula = unmplyd ~ .,
  data    = train_x,
  method  = "anova",
  control = list(minsplit = vals[1], maxdepth = vals[2], cp = vals[3])
)
pred <- predict(m3, newdata = test_x)
RMSE(pred = pred, obs = test_x$unmplyd)

# Try running the code below, using different values of the `prop` parameter passed to `split` (e.g. 0.6, 0.65, 0.7, 0.75, 0.8) to see the impact that differences in the split can have on the model (look at the internal and terminal nodes):
train.index = createDataPartition(anal_data$unmplyd, p = 0.7, list = F)
train_x2 = anal_data[train.index,] 
test_x2 = anal_data[-train.index,] 
m.temp <- rpart(
  formula = unmplyd ~ .,
  data    = train_x2,
  method  = "anova"
)
m.temp$cptable
rpart.plot(m.temp)

## 4. Bagging: Bootstrap aggregating

# a 10-fold cross-validated bagged model is constructed: 
ctrl <- trainControl(method = "cv",  number = 20) 
# CV bagged model
bagged_cv <- train(
  form = unmplyd ~ .,
  data    = train_x,
  method  = "treebag",
  trControl = ctrl,
  importance = TRUE
)
bagged_cv

tmp = varImp(bagged_cv)
data.frame(name = rownames(tmp$importance), 
           value = tmp$importance$Overall)  %>% 
  arrange(desc(value)) %>%
  ggplot(aes(reorder(name, value), value)) +
  geom_col(fill = "thistle") +
  coord_flip() + 
  xlab("") +theme_bw()

pred <- predict(bagged_cv, test_x)
RMSE(pred, test_x$unmplyd)


## 5. Random Forests

# the code below implements the RF function from the `randomForest` package:
# set random seed for reproducibility
set.seed(789)
# default RF model
m4 <- randomForest(
  formula = unmplyd ~ .,
  data = train_x
)
m4
pred <- predict(m4, test_x)
RMSE(pred, test_x$unmplyd)

# Plotting the error rate
ggplot(data.frame(error = m4$mse), aes(y = error, x = 1:500)) + 
  geom_line()+
  xlab("trees")+theme_bw()

which.min(m4$mse)
sqrt(m4$mse[which.min(m4$mse)])

# measure predictive accuracy if we do not want to use the OOB samples by splitting the training data to create a second training and validation set (xtest and ytest):
# create training and validation data 
# set random seed for reproducibility
set.seed(789)
train.index = createDataPartition(anal_data$unmplyd, p = 0.7, list = F)
train_x2 = anal_data[train.index,] 
test_x2 = anal_data[-train.index,] 
x_test <- train_x2[setdiff(names(train_x2), "unmplyd")]
y_test <- train_x2$unmplyd
rf_comp <- randomForest(
  formula = unmplyd ~ .,
  data    = train_x2,
  xtest   = x_test,
  ytest   = y_test
)
# extract OOB & validation errors
oob <- sqrt(rf_comp$mse)
validation <- sqrt(rf_comp$test$mse)


# the OOB and new split error rates can be visualised as in 
tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_comp$ntree) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  xlab("Number of trees")

params <- expand.grid(
  mtry       = seq(2, 7),
  node_size  = seq(3, 9, by = 2),
  samp_size = c(.60, .65, .70, .75, .80)
)
dim(params)

# ranger implementation random forest
res.vec = vector()
for(i in 1:nrow(params)) {
  # create the model
  m.i <- ranger(
    formula         = unmplyd ~ ., 
    data            = train_x, 
    num.trees       = 500,
    mtry            = params$mtry[i],
    min.node.size   = params$node_size[i],
    sample.fraction = params$samp_size[i],
    seed            = 123
  )
  # add OOB error to res.vec
  res.vec <- c(res.vec, sqrt(m.i$prediction.error))
  # to see progress
  if (i%%10 == 0) cat(i, "\t")
}
# best performing can be examined
params$OOB = res.vec
head(params[order(params$OOB),])

RMSE.vec <- vector()
for(i in 1:100) {
  m.i <- ranger(
    formula         = unmplyd ~ ., 
    data            = train_x, 
    num.trees       = 500,
    mtry            = 5,
    min.node.size   = 3,
    sample.fraction = .65,
    importance      = 'impurity'
  )
  RMSE.vec <- c(RMSE.vec, sqrt(m.i$prediction.error))
  if (i%%10 == 0) cat(i, "\t")
}

ggplot(data.frame(rmse = RMSE.vec), aes(x = rmse))+
  geom_histogram(bins = 15, fill = "tomato2", col = "grey")

# Variables that associated with the reducted random forest errors
data.frame(name = names(m.i$variable.importance), 
           value = m.i$variable.importance)  %>% 
  arrange(desc(value)) %>%
  ggplot(aes(reorder(name, value), value)) +
  geom_col(fill = "dodgerblue") +
  coord_flip() +
  xlab("")

## 6. Summary

## References