library(sf)
library(tidyverse)
library(tmap)
library(rpart)
library(rpart.plot)
library(visNetwork)
library(caret)
library(randomForest)
library(ranger)
library(sparkline)

load("Assignment_prep.RData")
oa_sf

oa_sf %>% st_drop_geometry() %>%
  select(-c(code, Easting, Northing)) -> anal_data
# create training and testing data
set.seed(1234) # for reproducibility0

#caret package
train.index = createDataPartition(anal_data$unmplyd, p = 0.7, list = F)
# split into training and validation (testing) subsets:
train_x = anal_data[train.index,]
test_x = anal_data[-train.index,]

#You can examine the distributions of the target variable
summary(anal_data$unmplyd[train.index])
summary(anal_data$unmplyd[-train.index])

#A regression tree can be fitted tree using the rpart package:
set.seed(1234)
m1 <- rpart(
  formula = unmplyd ~ .,
  data = train_x,
  method = "anova"
)

m1
rpart.plot(m1, box.palette = "Or") 
#shows the percentage of data each node and the average unmployd
#percentages for that branch.

#more refined version
visTree(m1,legend=FALSE,collapse=TRUE,direction='LR')
#This identifies the variable and threshold values
#that reduces the SSE - the Sum of Squares Error - in this case the difference between observed values
#of unmplyd and the potential group mean of the branch.
#The predictor variables have been partitioned in a top-down, greedy way

plotcp(m1)
#This shows the error (y-axis) and cost complexity (x-axis) and in this case there are diminishing returns after 7 terminal nodes
#So here a tree with 7 terminal nodes could be used with similar results.

round(m1$cptable, 4)

#Having the established the model and considered the cost complexity parameter (????), further fine tuning of
#2 rpart parameters can be undertaken.

#determine the minimum number of data points required
#to attempt a split before a terminal node is created
#the minsplit parameter, which has a default of 20.


#Second, the maximum number of internal nodes between the root and terminal node
#which has a default is 30 (the maxdepth parameter)


set.seed(1234)
m2 <- rpart(
  formula = unmplyd ~ .,
  data = train_x,
  method = "anova",
  control = list(minsplit = 30, maxdepth = 12)
)
m2$cptable

# More usefully a range of parameter values should be evaluated. 
#The code below creates a data frame of 189 combinations of minsplit and maxdepth parameters:
params <- expand.grid(
  minsplit = seq(10, 30, 1),
  maxdepth = seq(4, 12, 1)
)

dim(params)

head(params)


#Each combination of these can be passed to rpart in a loop to generate a table of results:
param_test <- matrix(nrow = 0, ncol = 4)
colnames(param_test) = c("minsplit", "maxdepth", "cp", "xerror")
for (i in 1:nrow(params)) {
  # get values for row i in params
  minsplit.i <- params$minsplit[i]
  maxdepth.i <- params$maxdepth[i]
  # create the model
  mod.i <- rpart(
    formula = unmplyd ~ .,
    data = train_x,
    method = "anova",
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
  # if (i%%10 == 0) cat(i, "\t")
}
param_test = data.frame(param_test)

#The table of results can be ordered and displayed:
head(param_test[order(param_test$xerror),])

#the code below creates a model using these parameters and uses this to make predictions
#for the test_x data created at the start of this:
set.seed(1234)
# assign the best to a vector
vals = param_test[order(param_test$xerror)[1],]
m3 <- rpart(
  formula = unmplyd ~ .,
  data = train_x,
  method = "anova",
  control = list(minsplit = vals[1], maxdepth = vals[2], cp = vals[3])
)
pred <- predict(m3, newdata = test_x)
RMSE(pred = pred, obs = test_x$unmplyd)
#Here the optimal model predicts the proportion of unemployment in any given LSOA from the variables in
#the social data table within 5.19, 1'% of the actual value

#using different values of the prop parameter passed to split
train.index = createDataPartition(anal_data$unmplyd, p = 0.7, list = F)
train_x2 = anal_data[train.index,]
test_x2 = anal_data[-train.index,]
m.temp <- rpart(
  formula = unmplyd ~ .,
  data = train_x2,
  method = "anova"
)
m.temp$cptable
rpart.plot(m.temp)


#In order to overcome the high variance problem 
#of single regression trees described above, 
#Bootstrap aggregating (bagging) was proposed
#to improve regression tree performance.

#Bagging, as the name suggests generates multiple 
#models with the same parameters and averages the results from multiple trees.

#Three steps to Bagging:
#1. Create a number of samples from the training data.
#These are termed Bootstrapped samples, 
#2. For each bootstrap sample create (train) a regression tree.
#3. Determine the average predictions from each tree, to generate an overall average predicted value


#The code below illustrates bagging using the caret package
ctrl <- trainControl(method = "cv", number = 20)
# CV bagged model
bagged_cv <- train(
  form = unmplyd ~ .,
  data = train_x,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE
)

bagged_cv

#The imporant of the different variables under a bagging approach
#in percentages
tmp = varImp(bagged_cv)
data.frame(name = rownames(tmp$importance),
           value = tmp$importance$Overall) %>%
  arrange(desc(value)) %>%
  ggplot(aes(reorder(name, value), value)) +
  geom_col(fill = "thistle") +
  coord_flip() +
  xlab("") +theme_bw()


pred <- predict(bagged_cv, test_x)
RMSE(pred, test_x$unmplyd)
# If this is compared to the test set out of sample, the cross-validated
#error estimate was very close, and the error has been reduced
#to about 4.64%


#Bagging regression trees results in a predictive model 
#that overcomes the problems of high variance

#problems with Bagging regression trees, which may exhibit tree correlation
#(i.e. the trees in the bagging process are not completely independent of 
#each other because all the predictors are considered at every split of every tree).

#As a result, trees from different bootstrap samples will have a similar structure 
#to each other and this can prevent Bagging from optimally reducing variance of 
#the prediction and the performance of the model.




###Random Forest

#seek to reduce tree correlation. They build large collections of decorrelated
#trees by adding randomness to the tree construction process.


#basic regression random forest algorithm proceeds as follows:
#1. Select the number of trees (`ntrees`)
#2. for i in `ntrees` do
#3. | Create a bootstrap sample of the original data
#4. | Grow a regression tree to the bootstrapped data
#5. | for each split do
#6. | | Randomly select `m` variables from `p` possible variables
#7. | | Identify the best variable/split among `m`
#8. | | Split into two child nodes
#9. | end
#10. end

set.seed(789)
# default RF model
m4 <- randomForest(
  formula = unmplyd ~ .,
  data = train_x
)
m4

pred <- predict(m4, test_x)
RMSE(pred, test_x$unmplyd)

#Plotting the error rate, as in figure 4 shows that it
#stabilizes at around 250 trees and slowly decreases to around 300 trees
ggplot(data.frame(error = m4$mse), aes(y = error, x = 1:500)) +
  geom_line()+
  xlab("trees")+theme_bw()

#The bootstrap sample in bagging will on average contain 63% of the 
#training data, with about 37% left out of the bootstrapped sample
#This is the out-of-bag (OOB) sample which are used to determine the
#model's accuracy through a cross-validation process.


##The error rate in the plot above is based on the OOB sample error (see m4$mse)
# we can determine the number of trees providing the lowest error rate
which.min(m4$mse)

sqrt(m4$mse[which.min(m4$mse)])


#We can measure predictive accuracy if we do not want to 
#use the OOB samples by splitting the training
#data to create a second training and validation set

## create training and validation data
# set random seed for reproducibility
set.seed(789)
train.index = createDataPartition(anal_data$unmplyd, p = 0.7, list = F)
train_x2 = anal_data[train.index,]
test_x2 = anal_data[-train.index,]
x_test <- train_x2[setdiff(names(train_x2), "unmplyd")]
y_test <- train_x2$unmplyd
rf_comp <- randomForest(
  formula = unmplyd ~ .,
  data = train_x2,
  xtest = x_test,
  ytest = y_test
)
# extract OOB & validation errors
oob <- sqrt(rf_comp$mse)
validation <- sqrt(rf_comp$test$mse)



##The OOB and new split error rates can be visualised 
tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_comp$ntree) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  xlab("Number of trees")

validation2 <- sqrt(rf_comp$test$mse[which.min(rf_comp$test$mse)]) 
validation2

#RMSE is reduced to well below 2.5% without any
# tuning which, which is much lower than the RMSE achieved with a 
#fully-tuned bagging model above

#Thevmodel can potentially be improved further by 
#additional tuning of the random forest model

#The main or initial consideration in tuning is to
#determine the number of candidate variables to select
#from at each split


## ntree number of trees
## mtry the number of variables to randomly sample at each split
# sampsize number of samples to train on, typically range 60-80%
# nodesize minimum number of samples in the terminal nodes and this defines the complexity of the trees.
# maxnodes sets the maximum number of terminal nodes 

params <- expand.grid(
  mtry = seq(2, 7),
  node_size = seq(3, 9, by = 2),
  samp_size = c(.60, .65, .70, .75, .80)
)
dim(params)

#below is same as above but meant to be quicker
res.vec = vector()
for(i in 1:nrow(params)) {
  # create the model
  m.i <- ranger(
    formula = unmplyd ~ .,
    data = train_x,
    num.trees = 500,
    mtry = params$mtry[i],
    min.node.size = params$node_size[i],
    sample.fraction = params$samp_size[i],
    seed = 123
  )
  # add OOB error to res.vec
  res.vec <- c(res.vec, sqrt(m.i$prediction.error))
  # to see progress
  if (i%%10 == 0) cat(i, "\t")
}

#The best performing can be examined
params$OOB = res.vec
head(params[order(params$OOB),])
  
#We can repeat this model to get a better expectation of the error rate.
RMSE.vec <- vector()
for(i in 1:100) {
  m.i <- ranger(
    formula = unmplyd ~ .,
    data = train_x,
    num.trees = 500,
    mtry = 5,
    min.node.size = 3,
    sample.fraction = .65,
    importance = 'impurity'
  )
  RMSE.vec <- c(RMSE.vec, sqrt(m.i$prediction.error))
  if (i%%10 == 0) cat(i, "\t")
}

ggplot(data.frame(rmse = RMSE.vec), aes(x = rmse))+
  geom_histogram(bins = 15, fill = "tomato2", col = "grey")
#In this case, the expected error ranges between 
#~4.26%-4.31% with probably value around 4.29%


data.frame(name = names(m.i$variable.importance),
           value = m.i$variable.importance) %>%
  arrange(desc(value)) %>%
  ggplot(aes(reorder(name, value), value)) +
  geom_col(fill = "dodgerblue") +
  coord_flip() +
  xlab("")
#impurity allows variable importance to be assessed
# a measure of the decrease in MSE each time a variable is used as a node split in a tree
# we can see that Degree has the greatest impact 
#in reducing MSE across the trees, followed by OAC_class


