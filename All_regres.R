##libraries import
library(car)
library(randomForest)
library(caret)
library(corrplot)
library(corrgram)
library(caTools)
library(Metrics)
library(dplyr)
library(glmnet)
library(e1071)
library(xgboost)
library(tidyr)
library(pls)
library(tibble)
library(ggplot2)
library(plotly)


##Function - Linear regression
fn_linear_model <- function(final_df){
  ## train test split
  
  final_df_reduced <- final_df[,-c(3,6,9,12,15,16,34,36,37,46,47,49)]
  set.seed(123)
  sample = sample.split(final_df_reduced$`Output_variable`, SplitRatio = 0.8)
  train = subset(final_df_reduced, sample == TRUE)
  test = subset(final_df_reduced, sample == FALSE)
  
  set.seed(123)
  lm_model_final <-  lm(`Output_variable` ~ ., train)
  summary(lm_model_final)
  
  pred_val <- predict(lm_model_final, test)
  
  #Accuracy check
  rmse_val <- rmse(test$`Output_variable`, pred_val) #   0.543476
  
  actuals_preds <- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pred_val))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds)  # 14.6%
  #head(actuals_preds)
  
  metrics_val <- postResample(pred_val, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  ##Predicting on complete dataset
  pred_val_all <- predict(lm_model_final, final_df)
  
  return(list(metrics_val,cor_val,pred_val_all))
}



##Function - Linear regression using vif reduced variables
fn_linear_reduced <- function(final_df){
  
  #final_df_reduced <- final_df[,-c(4,7,10,13,14)]
  final_df_reduced <- final_df[,-c(3,6,9,12,15,16,34,36,37,46,47,49)]
  
  set.seed(123)
  lm_model_final <-  lm(`Output_variable` ~., final_df_reduced)
  summary(lm_model_final)
  
  ## train test split
  set.seed(123)
  sample = sample.split(final_df$`Output_variable`, SplitRatio = 0.8)
  train = subset(final_df, sample == TRUE)
  test = subset(final_df, sample == FALSE)
  
  
  
  #vif
  vif_df <- as.data.frame(vif(lm_model_final))
  vif_df$Variable <- rownames(vif_df) 
  rownames(vif_df) <-NULL
  names(vif_df)[1] <- "vif_value" # all are <10, so critical
  
  reduced_vars <- vif_df$Variable[vif_df$vif_value<=15]
  vif_df[vif_df$Variable %in% reduced_vars,]
  
  ##Linear regression using reduced variables
  reduced_vars_merged <- paste0(reduced_vars, collapse = '+')
  
  set.seed(123)
  linear_model_red <- eval(parse(text = paste0("lm(`Output_variable` ~ ",reduced_vars_merged,",data = train)")))
  summary(linear_model_red)
  
  pred_val <- predict(linear_model_red, test)
  
  #Accuracy check
  rmse_val <- rmse(test$`Output_variable`, pred_val) #   0.3778134
  
  actuals_preds <- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pred_val))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds)  # 43%
  head(actuals_preds)
  metrics_val <- postResample(pred_val, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(linear_model_red, final_df_reduced)
  
  return(list(metrics_val,cor_val,pred_val_all, reduced_vars))
}



##Function - Linear regression using variables which are having correlation >0.3 and <=-0.3
fn_linear_correlated <- function(final_df){
  
  final_df_reduced <- final_df[,-c(3,6,9,12,15,16,34,36,37,46,47,49)]
  
  ## train test split
  set.seed(123)
  sample = sample.split(final_df_reduced$`Output_variable`, SplitRatio = 0.8)
  train = subset(final_df_reduced, sample == TRUE)
  test = subset(final_df_reduced, sample == FALSE)
  
  cor.data <- cor(final_df_reduced[,colnames(final_df_reduced)]) 
  corr_table <- as.data.frame(cor.data)
  corr_vars<- row.names(corr_table[corr_table$`Output_variable`>=0.15 | corr_table$`Output_variable`<= -0.15,])
  corr_vars <- corr_vars[corr_vars!="NA"]
  
  #corrplot(cor(final_df) ,method="color")
  #corrgram(final_df,order=TRUE, lower.panel=panel.shade,
  #        upper.panel=panel.pie, text.panel=panel.txt)
  
  train_corr <- train[,corr_vars]
  test_corr <- test[,corr_vars]
  
  set.seed(123)
  lm_corr_model <- lm(`Output_variable` ~ ., data = train_corr)
  summary(lm_corr_model)
  
  pred_val <- predict(lm_corr_model, test_corr)
  
  #Accuracy check
  rmse_val <- rmse(test_corr$`Output_variable`, pred_val) #   0.3778134
  
  actuals_preds <- data.frame(cbind(actuals=test_corr$`Output_variable`, predictedval=pred_val))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds)  # 43%
  head(actuals_preds)
  metrics_val <- postResample(pred_val, test_corr$`Output_variable`)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(lm_corr_model, final_df)
  
  return(list(metrics_val,cor_val,pred_val_all, corr_vars))
}




##Function - stepwise linear regression
fn_stepwise <- function(final_df){
  
  final_df_reduced <- final_df[,-c(3,6,9,12,15,16,34,36,37,46,47,49)]
  ## train test split
  set.seed(123)
  sample = sample.split(final_df_reduced$`Output_variable`, SplitRatio = 0.8)
  train = subset(final_df_reduced, sample == TRUE)
  test = subset(final_df_reduced, sample == FALSE)
  
  set.seed(123)
  lm_model_final <-  lm(`Output_variable` ~., final_df_reduced)
  summary(lm_model_final)
  
  set.seed(123)
  lm_step_model <-  step(lm_model_final, train)
  summary(lm_step_model)
  pred_val_step <- predict(lm_step_model, newdata = test)
  summary(lm_step_model)
  
  step_vars <- lm_step_model$call$formula
  
  #Accuracy check
  rmse_val <- rmse(test$`Output_variable`, pred_val_step) #   0.3778134
  
  actuals_preds <- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pred_val_step))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds)  # 43%
  head(actuals_preds)
  metrics_val <- postResample(pred_val_step, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(lm_step_model, final_df_reduced)
  
  return(list(metrics_val,cor_val,pred_val_all, lm_step_model))
}



##Function - stepwise linear regression with cross validation  10 folds
fn_stepwise_cv <- function(final_df){
  
  ## train test split
  set.seed(123)
  sample = sample.split(final_df$`Output_variable`, SplitRatio = 0.8)
  train = subset(final_df, sample == TRUE)
  test = subset(final_df, sample == FALSE)
  
  cctrl1 <- trainControl(method = "cv", number = 10, returnResamp = "all")
  
  
  
  set.seed(123)
  step_cv_model <- train(`Output_variable` ~ ., data = train, 
                         method = "glmStepAIC", 
                         trControl = cctrl1,
                         trace = 0)
  step_cv_var <- names(step_cv_model$finalModel$coefficients)[2:length(step_cv_model$finalModel$coefficients)]
  
  pred_step_cv <- predict(step_cv_model, test)
  
  #Accuracy check
  rmse_val <- rmse(test$`Output_variable`, pred_step_cv) #  0.3255085
  
  actuals_preds_step_cv <- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pred_step_cv))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_step_cv)  #60%
  head(actuals_preds_step_cv)
  
  
  metrics_val <- postResample(pred_step_cv, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(step_cv_model, final_df)
  
  return(list(metrics_val,cor_val,pred_val_all, step_cv_var))
}



##Function - Random forest

fn_randomforest <- function(final_df){
  
  final_df_rf <- final_df
  names(final_df_rf) <- gsub(" ",".",names(final_df_rf))
  
  ## train test split
  set.seed(123)
  sample = sample.split(final_df_rf$Output_variable, SplitRatio = 0.8)
  train = subset(final_df_rf, sample == TRUE)
  test = subset(final_df_rf, sample == FALSE)
  
  rf_model_final <-  randomForest(Output_variable ~ ., data=train, importance = TRUE)
  rf_model_final
  
  pred_val_rf <- predict(rf_model_final, newdata = test)
  
  #Accuracy check
  rmse_val <- rmse(test$Output_variable, pred_val_rf) #  0.3255085
  
  actuals_preds_rf <- data.frame(cbind(actuals=test$Output_variable, predictedval=pred_val_rf))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_rf)  #60%
  head(actuals_preds_rf)
  
  
  metrics_val <- postResample(pred_val_rf, test$Output_variable)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(rf_model_final, final_df_rf)
  
  return(list(metrics_val,cor_val,pred_val_all))
}



##Function - Random forest - Top 20
topn <- 20

fn_randomforest_topn <- function(final_df){
  
  final_df_rf <- final_df
  names(final_df_rf) <- gsub(" ",".",names(final_df_rf))
  
  rf_model <-  randomForest(Output_variable ~ ., data= final_df_rf, importance = TRUE)
  rf_model
  
  varImpPlot(rf_model) 
  importance(rf_model)
  varImp(rf_model)
  
  rf_varImp <- varImp(rf_model)
  rf_varImp$Variable <- row.names(rf_varImp)
  row.names(rf_varImp) <- NULL
  rf_varImp <- rf_varImp[order(rf_varImp$Overall, decreasing = TRUE),]
  Topn_rf_varImp <- rf_varImp[1:topn,]
  
  rf_Topn_df <-  final_df_rf[,c(Topn_rf_varImp$Variable,'Output_variable')]
  
  set.seed(123)
  sample <- sample.split(rf_Topn_df$Output_variable,SplitRatio = 0.8)  # any column can be given, just as standard response variable is given.
  train <- subset(rf_Topn_df,sample==TRUE)
  test <- subset(rf_Topn_df,sample==FALSE)
  
  rf_model_final <-  randomForest(Output_variable ~ ., data=train, importance = TRUE)
  rf_model_final
  
  pred_val_rf <- predict(rf_model_final, newdata = test)
  
  #Accuracy check
  rmse_val <- rmse(test$Output_variable, pred_val_rf) #  0.3255085
  
  actuals_preds_rf <- data.frame(cbind(actuals=test$Output_variable, predictedval=pred_val_rf))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_rf)  #60%
  head(actuals_preds_rf)
  
  
  metrics_val <- postResample(pred_val_rf, test$Output_variable)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(rf_model_final, final_df_rf)
  
  return(list(metrics_val,cor_val,pred_val_all,Topn_rf_varImp$Variable))
}



##function - Ridge Regression
fn_ridge <- function(final_df){
  
  #Train test split
  set.seed(123)
  sample <- sample.split(final_df$`Output_variable`,SplitRatio = 0.8)  # any column can be given, just as standard response variable is given.
  train <- subset(final_df,sample==TRUE)
  test <- subset(final_df,sample==FALSE)
  
  x = model.matrix(`Output_variable`~., data = train)[,-c(ncol(train))]
  y = train$`Output_variable`
  
  x_test = model.matrix(`Output_variable`~., data=test)[,-c(ncol(train))]
  y_test <- test$`Output_variable`
  
  grid = 10^ seq(10,-2,length = 100)
  
  ridge_model <- glmnet(x,y, alpha =0, lambda = grid)
  summary(ridge_model)
  
  #cross validation
  cv.out <- cv.glmnet(x,y,alpha = 0)
  bestlam_ridge = cv.out$lambda.min #0.2673379
  pred_ridge = predict(ridge_model, s =bestlam_ridge, newx = x_test)
  
  #Accuracy check
  rmse_val <- rmse(y_test, pred_ridge) #0.311811
  
  actuals_preds_ridge <- data.frame(cbind(actuals=y_test, predictedval=pred_ridge))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_ridge)  # 64.8%
  
  metrics_val <- postResample(pred_ridge, y_test)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(ridge_model, s =bestlam_ridge, newx = (model.matrix(`Output_variable`~., data=final_df)[,-c(ncol(final_df))]))
  
  return(list(metrics_val,cor_val,pred_val_all))
}




##Function - Ridge regression with Top 20 variables
fn_ridge_topn <- function(final_df){
  
  final_df_ridge <- final_df
  names(final_df_ridge) <- gsub(" ",".", names(final_df_ridge))
  
  x = model.matrix(Output_variable~., data = final_df_ridge)[,-c(ncol(final_df_ridge))]
  y = final_df_ridge$Output_variable
  
  grid = 10^ seq(10,-2,length = 100)
  
  #ridge_model <- glmnet(x,y, alpha =0, lambda = grid)
  #summary(ridge_model)
  
  #cross validation
  cv.out <- cv.glmnet(x,y,alpha = 0)
  bestlam_ridge = cv.out$lambda.min #0.2673379
  out_ridge = glmnet(x,y, alpha = 0, lambda = grid)
  ridge_coeff = predict(out_ridge, type= "coefficients", s= bestlam_ridge)
  ridge_coeff <- as.data.frame(as.matrix(ridge_coeff))
  ridge_coeff$variable <- rownames(ridge_coeff)
  rownames(ridge_coeff) <- NULL
  ridge_coeff <- ridge_coeff[(ridge_coeff$`1` !=0) & (!grepl("Intercept",ridge_coeff$variable)),]
  
  ridge_coeff_sorted <- ridge_coeff[order(ridge_coeff$`1`, decreasing = TRUE),]
  ridge_vars <- ridge_coeff_sorted$variable[1:topn]
  
  ridge_Topn_df <-  final_df_ridge[,c(ridge_vars,"Output_variable")]
  
  set.seed(123)
  sample <- sample.split(ridge_Topn_df$Output_variable,SplitRatio = 0.8)  # any column can be given, just as standard response variable is given.
  train <- subset(ridge_Topn_df,sample==TRUE)
  test <- subset(ridge_Topn_df,sample==FALSE)
  
  x = model.matrix(Output_variable ~ ., data = train)[,-c(ncol(train))]
  y = train$Output_variable
  
  x_test = model.matrix(Output_variable~., data=test)[,-c(ncol(test))]
  y_test <- test$Output_variable
  
  grid = 10^ seq(10,-2,length = 100)
  
  ridge_model <- glmnet(x,y, alpha =0, lambda = grid)
  summary(ridge_model)
  
  #cross validation
  cv.out <- cv.glmnet(x,y,alpha = 0)
  bestlam_ridge = cv.out$lambda.min # 0.2728557
  pred_ridge = predict(ridge_model, s =bestlam_ridge, newx = x_test)
  
  #Accuracy check
  rmse_val <- rmse(y_test, pred_ridge) #0.311811
  
  actuals_preds_ridge <- data.frame(cbind(actuals=y_test, predictedval=pred_ridge))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_ridge)  # 64.8%
  
  metrics_val <- postResample(pred_ridge, y_test)
  names(metrics_val) <- NULL
  
  #Prediction on all
  x = model.matrix(Output_variable ~ ., data = ridge_Topn_df)[,-c(ncol(ridge_Topn_df))]
  pred_val_all <- predict(ridge_model, s =bestlam_ridge, newx = x)
  
  return(list(metrics_val,cor_val,pred_val_all,ridge_vars))
  
}


##Function - Lasso regression
fn_lasso <- function(final_df){
  #Train test split
  set.seed(123)
  sample <- sample.split(final_df$`Output_variable`,SplitRatio = 0.8)  # any column can be given, just as standard response variable is given.
  train <- subset(final_df,sample==TRUE)
  test <- subset(final_df,sample==FALSE)
  
  x = model.matrix(`Output_variable` ~., data = train)[,-c(ncol(train))]
  y = train$`Output_variable`
  
  x_test = model.matrix(`Output_variable`~., data=test)[,-c(ncol(test))]
  y_test <- test$`Output_variable`
  
  grid = 10^ seq(10,-2,length = 100)
  
  lasso_model <- glmnet(x,y, alpha =1 , lambda = grid)
  set.seed(101)
  
  #cross validation
  cv.out <- cv.glmnet(x,y,alpha = 1)
  bestlam_lasso = cv.out$lambda.min #0.2673379
  pred_lasso= predict(lasso_model, s =bestlam_lasso, newx = x_test)
  
  #Accuracy check
  rmse_val <- rmse(y_test, pred_lasso) #0.311811
  
  actuals_preds_lasso <- data.frame(cbind(actuals=y_test, predictedval=pred_lasso))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_lasso)  # 64.8%
  
  metrics_val <- postResample(pred_lasso, y_test)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(lasso_model, s =bestlam_lasso, newx = (model.matrix(`Output_variable`~., data=final_df)[,-c(ncol(final_df))]))
  
  return(list(metrics_val,cor_val,pred_val_all))
  
}




##Function - SVM 
fn_svm <- function(final_df){
  ### train test split
  set.seed(123)
  sample <- sample.split(final_df$`Output_variable`,SplitRatio = 0.8)  
  train <- subset(final_df,sample==TRUE)
  test <- subset(final_df,sample==FALSE)
  
  set.seed(123)
  svm_model <- svm(`Output_variable` ~ ., data = train, scale = TRUE)
  svm_pred <- predict(svm_model, newdata = test)
  
  #Accuracy check
  rmse_val <- rmse(test$`Output_variable`, svm_pred) #0.311811
  
  actuals_preds_svm <- data.frame(cbind(actuals=test$`Output_variable`, predictedval=svm_pred))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_svm)  # 64.8%
  
  metrics_val <- postResample(svm_pred, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(svm_model, final_df[,-ncol(final_df)])
  
  return(list(metrics_val,cor_val,pred_val_all))
  
}



##Function - Gradient Boosting
fn_GradientBoosting <- function(final_df){
  
  ### train test split
  set.seed(123)
  sample <- sample.split(final_df$`Output_variable`,SplitRatio = 0.8)  
  train <- subset(final_df,sample==TRUE)
  test <- subset(final_df,sample==FALSE)
  
  set.seed(123)
  trainCtrl <- trainControl(method='repeatedcv', number = 5, repeats = 3)
  gbm_model <- train(`Output_variable` ~., data = train, method = 'gbm', distribution='gaussian',
                     trControl=trainCtrl, verbose = TRUE)
  
  gbm_pred <- predict(gbm_model, test)
  
  #Accuracy check
  
  rmse_val <- rmse(test$`Output_variable`, gbm_pred) # 0.3136362
  
  actuals_preds_gbm<- data.frame(cbind(actuals=test$`Output_variable`, predictedval=gbm_pred))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_gbm)  # 65%
  head(actuals_preds_gbm)
  
  metrics_val <- postResample(gbm_pred, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  #Prediction on all
  pred_val_all <- predict(gbm_model, final_df[,-ncol(final_df)])
  
  return(list(metrics_val,cor_val,pred_val_all))
}



##Function - Tuning gbm parameters
fn_gbm_tuned <- function(final_df){
  ### train test split
  set.seed(123)
  sample <- sample.split(final_df$`Output_variable`,SplitRatio = 0.8)  
  train <- subset(final_df,sample==TRUE)
  test <- subset(final_df,sample==FALSE)
  
  myGrid <- expand.grid(n.trees = c(150,175,200,225),
                        interaction.depth =c(5,6,7,8,9),
                        shrinkage = c(0.075,0.1,0.125,0.15,0.2),
                        n.minobsinnode=c(7,10,12,15))
  
  gbm_model_tuned <- train(`Output_variable` ~., data = train, method = 'gbm', distribution='gaussian',
                           trControl=trainCtrl, verbose = TRUE, tuneGrid = myGrid)
  
  ## finding best tune parameters
  gbm_model_tuned$bestTune 
  myGrid <- gbm_model$bestTune
  gbm_model_final <- train(`Output_variable` ~., data = train, method = 'gbm', distribution='gaussian',
                           trControl=trainCtrl, verbose = TRUE, tuneGrid = myGrid)
  
  gbm_pred <- predict(gbm_model_final, test)
  
  #Accuracy check
  
  rmse_val <- rmse(test$`Output_variable`, gbm_pred) # 0.2972779
  
  actuals_preds_gbm<- data.frame(cbind(actuals=test$`Output_variable`, predictedval=gbm_pred))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_gbm)  #68%
  head(actuals_preds_gbm)
  
  metrics_val <- postResample(gbm_pred, test$`Output_variable`)
  names(metrics_val) <-NULL
  
  #Prediction on all
  pred_val_all <- predict(gbm_model_final, final_df[,-ncol(final_df)])
  
  return(list(metrics_val,cor_val,pred_val_all))
}



##function - extreme Gradient Boosting
fn_xgboost <- function(final_df){
  ### train test split
  set.seed(123)
  sample <- sample.split(final_df$`Output_variable`,SplitRatio = 0.8)  
  train <- subset(final_df,sample==TRUE)
  test <- subset(final_df,sample==FALSE)
  
  train_label <- train[,"Output_variable"]
  train_matrix <- xgb.DMatrix(data = as.matrix(train[,-ncol(train)]), label = train_label)
  
  test_label <- test[,"Output_variable"]
  test_matrix <- xgb.DMatrix(data = as.matrix(test[,-ncol(test)]), label = test_label)
  
  # Parameters
  
  xgb_params <- list("objective" = "reg:linear",
                     "eval_metric" = "rmse"
  )
  watchlist <- list(train = train_matrix, test = test_matrix)
  
  # eXtreme Gradient Boosting Model
  bst_model <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = 1000,
                         watchlist = watchlist,
                         eta = 0.01)
  
  
  # Training & test error plot
  e <- data.frame(bst_model$evaluation_log)
  plot(e$iter, e$train_rmse, col = 'blue')
  lines(e$iter, e$test_rmse, col = 'red')
  
  
  best_iter <- e[e$test_rmse == min(e$test_rmse),]
  
  # eXtreme Gradient Boosting Model
  bst_model <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = best_iter$iter,
                         watchlist = watchlist,
                         eta = 0.01)
  
  # Prediction & confusion matrix - test data
  pred_xgboost <- predict(bst_model, newdata = test_matrix)
  
  #Accuracy check
  
  rmse_val <- rmse(test$`Output_variable`, pred_xgboost) # 0.3511358
  
  actuals_preds_xgboost<- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pred_xgboost))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_xgboost)  # 60.9%
  #head(actuals_preds_xgboost)
  
  metrics_val <- postResample(pred_xgboost, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  ##predict all
  final_df_matrix <- xgb.DMatrix(data = as.matrix(final_df[,-ncol(final_df)]), label = final_df$`Output_variable`)
  
  pred_xgboost_all <- predict(bst_model, newdata = final_df_matrix)
  
  return(list(metrics_val,cor_val,pred_xgboost_all))
  
}




##Function - extreme Gradient Boosting - Top 20 variables

fn_xgboost_topn <- function(final_df){
  
  ### train test split
  set.seed(123)
  sample <- sample.split(final_df$`Output_variable`,SplitRatio = 0.8)  
  train <- subset(final_df,sample==TRUE)
  test <- subset(final_df,sample==FALSE)
  
  train_label <- train[,"Output_variable"]
  train_matrix <- xgb.DMatrix(data = as.matrix(train[,-ncol(train)]), label = train_label)
  
  test_label <- test[,"Output_variable"]
  test_matrix <- xgb.DMatrix(data = as.matrix(test[,-ncol(test)]), label = test_label)
  
  # Parameters
  
  xgb_params <- list("objective" = "reg:linear",
                     "eval_metric" = "rmse"
  )
  watchlist <- list(train = train_matrix, test = test_matrix)
  
  # eXtreme Gradient Boosting Model
  bst_model <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = 1000,
                         watchlist = watchlist,
                         eta = 0.01)
  
  
  # Training & test error plot
  e <- data.frame(bst_model$evaluation_log)
  plot(e$iter, e$train_rmse, col = 'blue')
  lines(e$iter, e$test_rmse, col = 'red')
  
  
  best_iter <- e[e$test_rmse == min(e$test_rmse),]
  
  # eXtreme Gradient Boosting Model
  bst_model <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = best_iter$iter[1],
                         watchlist = watchlist,
                         eta = 0.01)
  
  # Feature importance
  imp <- xgb.importance(colnames(train_matrix), model = bst_model)
  print(imp)
  xgb.plot.importance(imp[1:topn])
  
  ###Using only top 20 variables in extreme graident boosting
  xgb_Topn_var <- c(imp[1:topn,"Feature"])
  xgb_Topn_var <- xgb_Topn_var$Feature[!is.na(xgb_Topn_var$Feature)]
  
  xgb_Topn_df <- final_df[,c(as.vector(unlist(xgb_Topn_var)),'Output_variable')]
  
  ### train test split
  set.seed(123)
  sample <- sample.split(xgb_Topn_df$`Output_variable`,SplitRatio = 0.8)  # any column can be given, just as standard response variable is given. 
  train <- subset(xgb_Topn_df,sample==TRUE)
  test <- subset(xgb_Topn_df,sample==FALSE)
  
  train_label <- train[,"Output_variable"]
  train_matrix <- xgb.DMatrix(data = as.matrix(train[,-ncol(train)]), label = train_label)
  
  test_label <- test[,"Output_variable"]
  test_matrix <- xgb.DMatrix(data = as.matrix(test[,-ncol(test)]), label = test_label)
  
  # Parameters
  
  xgb_params <- list("objective" = "reg:linear",
                     "eval_metric" = "rmse"
  )
  watchlist <- list(train = train_matrix, test = test_matrix)
  
  # eXtreme Gradient Boosting Model
  bst_model <- xgb.train(params = xgb_params,
                         data = train_matrix,
                         nrounds = best_iter$iter,
                         watchlist = watchlist,
                         eta = 0.01)
  
  
  # Prediction & confusion matrix - test data
  pred_xgboost_topn <- predict(bst_model, newdata = test_matrix)
  
  #Accuracy check
  
  rmse_val <- rmse(test$`Output_variable`, pred_xgboost_topn) # 0.3267764
  
  actuals_preds_xgboost_topn <- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pred_xgboost_topn))  # make actuals_predicteds dataframe.
  cor_val <- cor(actuals_preds_xgboost_topn)  # 64%
  #head(actuals_preds_xgboost)
  
  
  metrics_val <- postResample(pred_xgboost_topn, test$`Output_variable`)
  names(metrics_val) <-NULL  
  
  ##predict all
  xgb_Topn_df_matrix <- xgb.DMatrix(data = as.matrix(xgb_Topn_df[,-ncol(xgb_Topn_df)]), label = xgb_Topn_df$`Output_variable`)
  
  pred_xgboost_all <- predict(bst_model, newdata = xgb_Topn_df_matrix)
  
  return(list(metrics_val,cor_val,pred_xgboost_all,xgb_Topn_var))
  
}




## Function -PLS
fn_pls <- function(final_df){
  ##train test split
  set.seed(123)
  sample= sample.split(final_df$`Output_variable`, SplitRatio = 0.8)
  train = final_df[sample==TRUE,]
  test = final_df[sample==FALSE,]
  
  pls_model <- plsr(`Output_variable` ~ . , ncomp=25, data= train, validation = "LOO")
  
  # Find the number of dimensions with lowest cross validation error
  cv = RMSEP(pls_model)
  best.dims = which.min(cv$val[estimate = "adjCV", , ]) - 1
  
  if(best.dims==0){
    best.dims=5
  }
  
  # Rerun the model
  pls_model = plsr(`Output_variable` ~ ., data = train, ncomp = best.dims)
  
  coefficients = coef(pls_model)
  sum.coef = sum(sapply(coefficients, abs))
  coefficients = coefficients * 100 / sum.coef
  coefficients = sort(coefficients[, 1 , 1])
  explvar(pls_model)
  
  ##Prediction
  pls_pred = predict(pls_model, test[,-ncol(test)], ncomp=best.dims)
  
  #Accuracy check
  rmse_val <- rmse(test$`Output_variable`, pls_pred) # 0.3626902
  
  actuals_preds_pls<- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pls_pred))  # make actuals_predicteds dataframe.
  cor_val<- cor(actuals_preds_pls)  # 58%
  head(actuals_preds_pls)
  
  metrics_val <- postResample(pls_pred, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  ##predict all
  pls_pred_all = predict(pls_model, final_df[-ncol(final_df)], ncomp=best.dims)
  
  return(list(metrics_val,cor_val,pls_pred_all))
}



## Function - PCR
fn_pcr <- function(final_df){
  ##train test split
  set.seed(123)
  sample= sample.split(final_df$`Output_variable`, SplitRatio = 0.8)
  train = final_df[sample==TRUE,]
  test = final_df[sample==FALSE,]
  
  pcr_model <- pcr(`Output_variable` ~., data = train, scale = FALSE, validation = "CV")
  summary(pcr_model)
  
  # Find the number of dimensions with lowest cross validation error
  cv = RMSEP(pcr_model)
  best.dims = which.min(cv$val[estimate = "adjCV", , ]) - 1
  
  if(best.dims==0){
    best.dims=5
  }
  
  pcr_pred <- predict(pcr_model, test, ncomp = best.dims)
  
  #Accuracy check
  
  rmse_val <- rmse(test$`Output_variable`, pcr_pred) #  0.3236529
  
  actuals_preds_pcr<- data.frame(cbind(actuals=test$`Output_variable`, predictedval=pcr_pred))  # make actuals_predicteds dataframe.
  cor_val<- cor(actuals_preds_pcr)  # 64%
  head(actuals_preds_pcr)
  
  metrics_val <- postResample(pcr_pred, test$`Output_variable`)
  names(metrics_val) <- NULL
  
  ##predict all
  pcr_pred_all = predict(pcr_model, final_df[-ncol(final_df)], ncomp=best.dims)
  
  return(list(metrics_val,cor_val,pcr_pred_all))
  
}

  
  lm_result <- fn_linear_model(final_df)
  
  Result_df<- data.frame("Actual" = final_df$`Output_variable`, "Linear_Reg" = lm_result[[3]])
  
  summary_df <- data.frame("Algorithm"="Linear Regression","Variables_used"= "All except DT variables", "RMSE"=lm_result[[1]][1],
                                   "Rsquared"=lm_result[[1]][2],"MAE"=lm_result[[1]][3], "Correlation_Accuracy"=lm_result[[2]][2])
  
  
  
  lm_red_result <- fn_linear_reduced(final_df)
  
  Result_df$Linear_Reg_reduced <- lm_red_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="Linear Regression Reduced using vif","Variables_used"= paste0(lm_red_result[[4]],collapse=','), "RMSE"=lm_red_result[[1]][1],
                                                             "Rsquared"=lm_red_result[[1]][2],"MAE"=lm_red_result[[1]][3], "Correlation_Accuracy"=lm_red_result[[2]][2]))
  
  
  lm_corr_result <- fn_linear_correlated(final_df)
  
  Result_df$Linear_Reg_Correlated <- lm_corr_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="Linear Regression using correlation","Variables_used"= paste0(lm_corr_result[[4]],collapse=','), "RMSE"=lm_corr_result[[1]][1],
                                                             "Rsquared"=lm_corr_result[[1]][2],"MAE"=lm_corr_result[[1]][3], "Correlation_Accuracy"=lm_corr_result[[2]][2]))
  
  
  lm_stepwise_result <- fn_stepwise(final_df)
  
  Result_df$Stepwise_Reg <- lm_stepwise_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="Stepwise Regression","Variables_used"= paste0(variable.names(lm_stepwise_result[[4]])[2:length(variable.names(lm_stepwise_result[[4]]))],collapse=','), "RMSE"=lm_stepwise_result[[1]][1],
                                                             "Rsquared"=lm_stepwise_result[[1]][2],"MAE"=lm_stepwise_result[[1]][3], "Correlation_Accuracy"=lm_stepwise_result[[2]][2]))
  
  
  
  lm_stepwise_cv_result <- fn_stepwise_cv(final_df)
  
  Result_df$Stepwise_CV <- lm_stepwise_cv_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="Stepwise CV Regression","Variables_used"= paste0(lm_stepwise_cv_result[[4]],collapse=','), "RMSE"=lm_stepwise_result[[1]][1],
                                                             "Rsquared"=lm_stepwise_cv_result[[1]][2],"MAE"=lm_stepwise_cv_result[[1]][3], "Correlation_Accuracy"=lm_stepwise_cv_result[[2]][2]))
  
  rf_result <- fn_randomforest(final_df)
  
  Result_df$Random_Forest <- rf_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="Random Forest","Variables_used"= "All", "RMSE"=rf_result[[1]][1],
                                                             "Rsquared"=rf_result[[1]][2],"MAE"=rf_result[[1]][3], "Correlation_Accuracy"=rf_result[[2]][2]))
  
  rf_result <- fn_randomforest_topn(final_df)
  
  Result_df$Random_Forest_Topn <- rf_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"=paste0("Random Forest - Top",topn),"Variables_used"= paste0(rf_result[[4]],collapse=","), "RMSE"=rf_result[[1]][1],
                                                             "Rsquared"=rf_result[[1]][2],"MAE"=rf_result[[1]][3], "Correlation_Accuracy"=rf_result[[2]][2]))
  
  
  
  
  ridge_result <- fn_ridge(final_df)
  
  Result_df$Ridge <- ridge_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="Ridge Regression","Variables_used"= "All", "RMSE"=ridge_result[[1]][1],
                                                             "Rsquared"=ridge_result[[1]][2],"MAE"=ridge_result[[1]][3], "Correlation_Accuracy"=ridge_result[[2]][2]))
  
  
  ridge_topn_result <- fn_ridge_topn(final_df)
  
  Result_df$Ridge_Topn <- ridge_topn_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"=paste0("Ridge Regression Top",topn),"Variables_used"= paste0(ridge_topn_result[[4]], collapse=","), "RMSE"=ridge_topn_result[[1]][1],
                                                             "Rsquared"=ridge_topn_result[[1]][2],"MAE"=ridge_topn_result[[1]][3], "Correlation_Accuracy"=ridge_topn_result[[2]][2]))
  
  
  lasso_result <- fn_lasso(final_df)
  
  Result_df$Lasso <- lasso_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="Lasso Regression","Variables_used"= "All", "RMSE"=lasso_result[[1]][1],
                                                             "Rsquared"=lasso_result[[1]][2],"MAE"=lasso_result[[1]][3], "Correlation_Accuracy"=lasso_result[[2]][2]))
  
  svm_result <- fn_svm(final_df)
  
  Result_df$SVM <- svm_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="SVM","Variables_used"= "All", "RMSE"=svm_result[[1]][1],
                                                             "Rsquared"=svm_result[[1]][2],"MAE"=svm_result[[1]][3], "Correlation_Accuracy"=svm_result[[2]][2]))
  
  
  #gbm_result <- fn_GradientBoosting(final_df)
  
  #Result_df$GBM <- gbm_result[[3]]
  
  #summary_df <- rbind(summary_df, data.frame("Algorithm"="GBM","Variables_used"= "All", "RMSE"=gbm_result[[1]][1],
  #                                                  "Rsquared"=gbm_result[[1]][2],"MAE"=gbm_result[[1]][3], "Correlation_Accuracy"=gbm_result[[2]][2]))
  
  #gbm_tuned_result <- fn_gbm_tuned(final_df)
  
  #Result_df$GBM_Tuned <- gbm_tuned_result[[3]]
  
  #summary_df <- rbind(summary_df, data.frame("Algorithm"="GBM_Tuned","Variables_used"= "All", "RMSE"=gbm_tuned_result[[1]][1],
  #                                                   "Rsquared"=gbm_tuned_result[[1]][2],"MAE"=gbm_tuned_result[[1]][3], "Correlation_Accuracy"=gbm_tuned_result[[2]][2]))
  
  
  xgboost_result <- fn_xgboost(final_df)
  
  Result_df$XGBOOST <- xgboost_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="XGBOOST","Variables_used"= "All", "RMSE"=xgboost_result[[1]][1],
                                                             "Rsquared"=xgboost_result[[1]][2],"MAE"=xgboost_result[[1]][3], "Correlation_Accuracy"=xgboost_result[[2]][2]))
  
  xgboost_topn_result <- fn_xgboost_topn(final_df)
  
  Result_df$XGBOOST_Topn <- xgboost_topn_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="XGBOOST_Top20","Variables_used"= paste0(xgboost_topn_result[[4]],collapse=","), "RMSE"=xgboost_topn_result[[1]][1],
                                                             "Rsquared"=xgboost_topn_result[[1]][2],"MAE"=xgboost_topn_result[[1]][3], "Correlation_Accuracy"=xgboost_topn_result[[2]][2]))
  
  
  pls_result <- fn_pls(final_df)
  
  Result_df$PLS <- pls_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="PLS","Variables_used"= "All", "RMSE"=pls_result[[1]][1],
                                                             "Rsquared"=pls_result[[1]][2],"MAE"=pls_result[[1]][3], "Correlation_Accuracy"=pls_result[[2]][2]))
  
  
  pcr_result <- fn_pcr(final_df)
  
  Result_df$PCR <- pcr_result[[3]]
  
  summary_df <- rbind(summary_df, data.frame("Algorithm"="PCR","Variables_used"= "All", "RMSE"=pcr_result[[1]][1],
                                                             "Rsquared"=pcr_result[[1]][2],"MAE"=pcr_result[[1]][3], "Correlation_Accuracy"=pcr_result[[2]][2]))
  
  
 