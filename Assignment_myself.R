library(tidyverse)
library(rgdal)
library(sf)
library(tmap)
library(randomForest)
library(ranger)
library(caret)
library(scales)
library(sparkline)
library(Metrics)


# read the data
listings = as_tibble(read.csv("listings.csv", stringsAsFactors = F))
str(listings)

listings[, c("id", "price", "beds","property_type",
             "bedrooms", "bathrooms", "room_type" )]


###Modelling and an initial OLS Regression model


#An initial standard linear regression model 
#after some cleaning of the price variable
#convert prices to numbers
dollar_to_number = function(x) {
  x = gsub("[\\$]", "", x)
  x = gsub(",", "", x)
  x = as.numeric(x)
  x}
listings$price = dollar_to_number(listings$price)
listings$cleaning_fee = dollar_to_number(listings$cleaning_fee)
# convert cleaning fee to a binary 
listings$cleaning_fee = (listings$cleaning_fee > 0 )+0
listings$cleaning_fee[is.na(listings$cleaning_fee)] = 0


#X number of people the rental accommodates (accommodates);
#X number of bathrooms (bathrooms);
#X whether there is cleaning fee or not (cleaning_fee = this variable will have to be manipulated);
#X whether the property type is a House or Other (the property_type will need to be manipulated);
#X whether the room type is Private or Shared (the room_type variable will need to be manipulated);
#X the distance to the city centre



#We can examine the outputs:
badols = listings %>%
  select(price, accommodates, bathrooms, cleaning_fee,property_type, room_type) %>%
  drop_na()

m2=(lm(price~accommodates+bathrooms+cleaning_fee+
             factor(property_type)+factor(room_type),
           data = listings[!is.na(listings$price),]))
?glm
ggplot(data.frame(Observed = badols$price,
                  Predicted = m2$fitted.values),
       aes(x = Observed, y = Predicted))+
  geom_point(size = 1, alpha = 0.5)+
  geom_smooth(method = "lm", col = "red")
###Data Pre-processing

#listings data are messy, need to be cleaned 
#and new variables need to be created

# convert price to numbers
listings$price = dollar_to_number(listings$price)
hist(listings$price, col = "salmon", breaks = 150)

#This has a classic skewed distribution and need to be transformed.
#usual routes are using logs or square roots. 
hist(log(listings$price), col = "cornflowerblue")
#Logs look to be a good fit

#get rid of any records that have a rental price of zero 
listings = listings[listings$price >0,]
listings$log_price = log(listings$price)
summary(listings$log_price)

#Next we can reduce some of the types variables
#Here it looks like Apartment and House are the types that
#are most interest. The rest can be put into Other:
index = listings$property_type == "House" 
listings$property_type[!index] = "Other"

#Then we can convert some of the variables to binary variables including 
#cleaning_fee as before as well as some others that might
#potentially be interesting:

# others
listings$property_type_House = (listings$property_type == "House")+0
listings$property_type_Other = (listings$property_type == "Other")+0
listings$room_type_Private_room = (listings$room_type == "Private room")+0
listings$room_type_Shared_room = (listings$room_type == "Shared room")+0


#fill gaps (empty values or NAs) in data by applying the median value
listings$bathrooms[is.na(listings$bathrooms)] = median(listings$bathrooms, na.rm = T)

median(listings$bathrooms)
#the distance to the city centre can be calculated. 
#code below creates a pint sf layer using the latitude and longitude of Manchester city centre:
manc_cc = st_as_sf(
  data.frame(city = "manchester", longitude = -2.23743,
             latitude = 53.48095),
  coords = c("longitude", "latitude"), crs = 4326)

# distance in km to the listings observations can be calculated from their locations
listings$ccdist <- as.vector(
  st_distance(st_as_sf(listings,coords = c("longitude", "latitude"),
                       crs = 4326) , manc_cc))/1000


# the variables can be allocated to a new data table and checked:
data_anal_me = listings[,c("log_price", "accommodates", "bathrooms", "cleaning_fee",
                        "property_type_House", "property_type_Other", "room_type_Private_room",
                        "room_type_Shared_room", "ccdist")]
summary(data_anal_me)
data_anal_me
?as.formula
?lm
###A refined OLS Regression initial model
#second OLS regression model can be fitted
#using the log of rental price as the target variable


reg.mod =
  as.formula(log_price ~ accommodates + bathrooms +
               cleaning_fee + property_type_House +
               property_type_Other + room_type_Private_room +
               room_type_Shared_room + ccdist)
m = lm(reg.mod, data = data_anal_me)
summary(m)
#much improved but probs some collinearity

#MAPE
MAPE

#Observed = badols$price,
#Predicted = m2$fitted.values

?mape
Metrics::mape(badols$price, m2$fitted.values)
Metrics::mape(data_anal_me$log_price, m$fitted.values)
Metrics::mape(test_x1$log_price, pred.rf)
#Predicted = pred.rf, Observed = test_x1$log_price

#The relative importance of each predictor variable can also be examined
varImp(m, C(0, 100))
??VarImp

#the model prediction compared with the actual, observed logged price values
data.frame(Observed = data_anal_me$log_price,
            Predicted = m$fitted.values)


ggplot(data.frame(Observed = data_anal_me$log_price,
                  Predicted = m$fitted.values),
       aes(x = Observed, y = Predicted))+
  geom_point(size = 1, alpha = 0.5)+
  geom_smooth(method = "lm", col = "red")



#the model can be applied to the test data to make 
#prediction of the likely AirBnb rental price for
#potentially new Airbnb properties. 
#Load the potential rentals listings and examine them:
load("potential_rentals.RData")
data.frame(potential_rentals)

#The regression model above can be used to predict 
#the market rate the predict function and the antilog function exp
predOLS = exp(predict(m, newdata = potential_rentals))
predOLS

?exp
?predict

#with a bit of cleaning, and rounding to $5, the recommended prices would be as below:
data.frame(ID = potential_rentals$ID,
           `OLS Price` = paste0("$", round(pred/5)*5))


data.frame(ID = potential_rentals$ID)


###Random Forest exp


#createDataPartition function form caret ensures that the
#target variable has the same distribution in the 
#training and validation (test) split:
set.seed(123) # reproducibility
train.index1 = createDataPartition(data_anal_me$log_price, p = 0.7, list = F)
summary(data_anal_me$log_price[train.index1])

summary(data_anal_me$log_price[-train.index1])

train_x1 = data_anal_me[train.index1,]
test_x1 = data_anal_me[-train.index1,]


#create intial model using the RF implementation 
#determine an appropriate number of trees 
reg.mod = log_price ~ .
rf1 <- randomForest(
  formula = reg.mod, ntree= 1000,
  data = train_x1
)
# number of trees with lowest error
which.min(rf1$mse) ### why is this saying interger zero???



# plot!
plot(rf1)


#set up a tuning grid
params <- expand.grid(
  mtry = c(2:8), # the max value should be equal to number of predictors
  node_size = seq(3, 9, by = 2),
  samp_size = c(0.6, 0.65, 0.7, 0.75, 0.8)
)

?mtry
# have a look!
dim(params)

head(params)

tail(params)

#a loop can be set up that passes each combination 
#of parameters in turn to the RF algorithm, with the
#error results saved off!
  
# define a vector to save the results of each iteration of the loop
rf.grid = vector()

# now run the loop
for(i in 1:nrow(params)) {
  # create the model
  rf.i <- ranger(
    formula = reg.mod,
    data = train_x1,
    num.trees = 750,
    mtry = params$mtry[i],
    min.node.size = params$node_size[i],
    sample.fraction = params$samp_size[i],
    seed = 123
  )
  # add OOB error to res.vec
  rf.grid <- c(rf.grid, sqrt(rf.i$prediction.error))
  # to see progress
  if (i%%10 == 0) cat(i, "\t")
}

# add the result to params
params$OOB = rf.grid

#Now the results can be inspected anf the best performing
#combination of parameters extracted using which.min:
params[which.min(params$OOB),]


#this can be assigned the best_vals and passed to a final model
best_vals = unlist(params[which.min(params$OOB),])
rfFit = ranger(
  formula = reg.mod,
  data = train_x1,
  num.trees = 750,
  mtry = best_vals[1],
  min.node.size = best_vals[2],
  sample.fraction = best_vals[3],
  seed = 123,
  importance = "impurity"
)

#The final model can be evaluated by using it to predict median 
#income values for the test data
pred.rf = predict(rfFit, data = test_x1)$predictions
postResample(pred = pred.rf, obs = test_x1$log_price)

#to get Rsquared comraed with all data


#The model accuracy evaluated using the postResample 
#function form the caret package:
postResample(pred = pred.rf, obs = test_x1$log_price)
#The R squared (????2) error tells us that the final
#model explains about 60% of the variation


#plot the variable importance
#shows how the variables are contributing to the model
data.frame(name = names(rfFit$variable.importance),
           value = rescale(rfFit$variable.importance, c(0,100))) %>%
  arrange(desc(value)) %>%
  ggplot(aes(reorder(name, value), value)) +
  geom_col() + coord_flip() + xlab("") +
  theme(axis.text.y = element_text(size = 7))

rfFit$variable.importance


data.frame(name = names(rfFit$variable.importance),
           value = rescale(rfFit$variable.importance, c(0,100))) 

#predicted and observed data can be compared graphically as before:
data.frame(Predicted = pred.rf, Observed = test_x1$log_price) %>%
  ggplot(aes(x = Observed, y = Predicted))+
  geom_point(size = 1, alpha = 0.5)+
  geom_smooth(method = "lm", col = "red")

#RF model can be used to predict median income over 
#unknown or future observations with the estimates of 
#the input parameters:
# create a data.frame

?dim

#I removed some jazz here that may be essential


pred.results = exp(predict(rfFit, data = potential_rentals)$predictions)
pred.results


#fix this
data.frame(ID = potential_rentals$ID,
           `RF Price` = paste0("$", round(pred.results/5)*5))




#look into this bit
# create a data.frame
test_df = data.frame(accommodates = seq(10, 50, 10),
                     bathrooms = seq(10, 50, 10),
                     cleaning_fee = seq(10, 50, 10),
                     property_type_House = seq(10, 50, 10),
                     property_type_Other = seq(10, 50, 10),
                     room_type_Private_room= seq(10, 50, 10),
                     room_type_Shared_room= seq(10, 50, 10),
                     ccdist= seq(10, 50, 10))
# predict median income ($1000s)
round(predict(rfFit, data = test_df)$predictions, 3)




####graph of two models skip


ggplot(data.frame(OLS = predols1$predOLS,
                  Randomforest = predrf1$pred.results),
       aes(x = OLS, y = Randomforest))+
  geom_point(size = 2, alpha = 0.5)+
  geom_abline(method = "lm")

geom_abline()?geom_smooth
  
predols1 = data.frame(predOLS)
predrf1 = data.frame(pred.results)
####
