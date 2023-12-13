#############
##LIBRARIES##
#############

library(tidymodels) 
library(tidyverse)
library(vroom) 
library(randomForest)
library(doParallel)
library(lightgbm)
library(themis)
library(bonsai)
conflicted::conflicts_prefer(yardstick::rmse)
conflicted::conflicts_prefer(yardstick::accuracy)
conflicted::conflicts_prefer(yardstick::spec)

####################
##WORK IN PARALLEL##
####################

all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)


########
##DATA##
########

my_data <- vroom("train.csv")
test_data <- vroom("test.csv")

my_data$Cover_Type <- as.factor(my_data$Cover_Type)


###
##EDA##
#######

DataExplorer::plot_missing(my_data)
DataExplorer::plot_histogram(my_data)
DataExplorer::plot_bar(bake_1)
ggplot(data=bake_1) + 
  geom_point(mapping=aes(x=Slope,y=Cover_Type))
ggplot(data=bake_1) + geom_mosaic(aes(x=product(color), fill=Cover_Type))


library(MASS)
library(bestNormalize)
library(data.table)
setDT(my_data)
setDT(test_data)

response_var <- 'Cover_Type'
col_to_exclude <- c('Id')
my_data[,(col_to_exclude) := NULL]


# Remove the complete NA columns
col_to_exclude <- c('Soil_Type7', 'Soil_Type15')
my_data[,(col_to_exclude) := NULL]
test_data[,(col_to_exclude) := NULL]

# Build the list
col_name_X <- setdiff(colnames(my_data), c(response_var,col_to_exclude))

# Set the output type
my_data[,Cover_Type := as.factor(Cover_Type)]



for(col_name in colnames(my_data)[1:11])
{
  print(BNobject <- bestNormalize(my_data[,get(col_name)], k=5, r=1, allow_orderNorm = TRUE, allow_exp = TRUE))
  print('===================================')
  print(col_name)
  print(class(BNobject$chosen_transform))
  print('===================================')
}

col_name_transform_set <- c()

#============ for Log transforms
col_list_log_transform <- c(
  'Horizontal_Distance_To_Fire_Points',
  'Horizontal_Distance_To_Roadways'
)

for(col_name in col_list_log_transform)
{
  log_x_obj <- log_x(my_data[,get(col_name)])
  log_x_obj
  col_name_transform <- paste0(col_name,'_log')
  my_data[,(col_name_transform) := predict(log_x_obj, get(col_name))]
  test_data[,(col_name_transform) := predict(log_x_obj, get(col_name))]
  
  col_name_transform_set <- c(col_name_transform_set, col_name_transform)
}

head(my_data)
head(data_test)

##########
##RECIPE##
##########

my_recipe <- recipe(Cover_Type ~ ., data=my_data) %>%
  update_role(Id, new_role="id") %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_smote(all_outcomes(), neighbors=7)

prepped_recipe <- prep(my_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)



########
##BART##
########

bart_model <- bart(trees = 1113,
                   prior_terminal_node_coef = tune(),
                   prior_terminal_node_expo = tune(),
                   prior_outcome_range = tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(bart_model) 

tuning_grid_bart <- grid_regular(prior_terminal_node_coef(),
                                 prior_terminal_node_expo(),
                                 prior_outcome_range(),
                                 levels = 3)
folds_bart <- vfold_cv(my_data, v = 5, repeats=1)

CV_results_bart <- bart_workflow %>%
  tune_grid(resamples=folds_bart,
            grid=tuning_grid_bart,
            metrics=metric_set( f_meas, sens, recall,
                                accuracy))
bestTune_bart <- CV_results_bart %>%
  select_best("accuracy")

final_bart_wf <- bart_workflow %>% 
  finalize_workflow(bestTune_bart) %>% 
  fit(data = my_data)


bart_predictions<- final_bart_wf %>% 
  predict(new_data = test_data)


bart_prediction <- bind_cols(test_data$Id,bart_predictions$.pred_class)

colnames(bart_prediction) <- c("Id","Cover_Type")

bart_predictions <- as.data.frame(bart_predictions)

vroom_write(bart_prediction,"bart_predictions.csv",',')



############
##BOOSTING##
############

boost_model <- boost_tree(tree_depth=tune(),
                          trees= tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")

boost_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(boost_model) 

tuning_grid_boost <- grid_regular(tree_depth(),
                                  learn_rate(),
                                  trees(),
                                  levels = 5)
folds_boost <- vfold_cv(my_data, v = 10, repeats=1)

CV_results_boost <- boost_workflow %>%
  tune_grid(resamples=folds_boost,
            grid=tuning_grid_boost,
            metrics=metric_set(f_meas, sens, recall, accuracy))

bestTune_boost <- CV_results_boost %>%
  select_best("accuracy")

final_wf_boost <- boost_workflow %>% 
  finalize_workflow(bestTune_boost) %>% 
  fit(data = my_data)


boost_prediction <- final_wf_boost %>% 
  predict(new_data = test_data, type="class")


boost_predictions <- bind_cols(test_data$Id,boost_prediction$.pred_class)

colnames(boost_predictions) <- c("Id","Cover_Type")

vroom_write(boost_predictions,"boost_predictions.csv",',')


###################
##NEURAL NETWORKS##
###################

nn_recipe <- recipe(Cover_Type~., data=my_data) %>%
  update_role(Id, new_role="id") %>%
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_smote(all_outcomes(), neighbors=7) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) 

prepped_recipe <- prep(nn_recipe, verbose = T)
bake_1 <- bake(prepped_recipe, new_data = NULL)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50, #or 100 or 2507
                activation="relu") %>%
  set_engine("keras", verbose=0) %>% 
  set_mode("classification")

nn_workflow <- workflow() %>% #Creates a workflow
  add_recipe(nn_recipe) %>% #Adds in my recipe
  add_model(nn_model) 

tuning_grid_nn <- grid_regular(hidden_units(range=c(1,100)),
                               levels=3)
folds_nn <- vfold_cv(my_data, v = 5, repeats=1)

tuned_nn <- nn_workflow %>%
  tune_grid(resamples=folds_nn,
            grid=tuning_grid_nn,
            metrics=metric_set(roc_auc, f_meas, sens, recall, spec,
                               precision, accuracy))


bestTune_nn <- tuned_nn %>%
  select_best("accuracy")

final_rf_wf <- nn_workflow %>% 
  finalize_workflow(bestTune_nn) %>% 
  fit(data = my_data)


nn_predictions <- final_rf_wf %>% 
  predict(new_data = test_data, type="class")

nn_predictions <- bind_cols(test_data$Id,nn_predictions$.pred_class)

colnames(nn_predictions) <-  c("Id","Cover_Type")

nn_predictions <- as.data.frame(nn_predictions)

vroom_write(nn_predictions,"nn_predictions.csv",',')


##IF IT DOESNT WORK RUN THIS PYTHON CODE IN THE TERMINAL##

#pip install --upgrade pip 

## For GPU users
#pip install tensorflow[and-cuda]
## For CPU users
#pip install tensorflow

## THEN RUN install_keras() in R Console ##


