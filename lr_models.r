# install.packages('reticulate')
library(stargazer)
library(reticulate)
library(dplyr)
library(tidyr)
library(caret)
library(ROCR)
library(car)
library(ggplot2)
pd <- import("pandas")

options(scipen=999)
options(max.print=999999)

load_data <- function(filename) {
  df <- pd$read_pickle(filename)
  df_n <- df %>%
    unnest_wider(tfidf_vectorized, names_sep = '_')
  df_n$bin_helpfulness <- ifelse(df_n$helpfulness == 'Helpful', 1, 0)
  df_final <- df_n[, !names(df_n) 
               %in% c('product_id', 'product_title', 'product_price', 
                      'user_id', 'username', 'review/helpfulness', 
                      'review_rating', 'review_summary', 
                      'review_text', 'helpful_votes', 'votes', 
                      'helpfulness', 'helpfulness_ratio', 
                      'tokenized_text', 'pre_processed_text', 'review_time', 
                      'avg_product_rating', 'richness', 'review_length', 'vectorized_text')]
  return(df_final)
}

summary_or <- function(m) {
  s <- summary(m)
  s$coefficients <- cbind(exp(s$coefficients[,1]), s$coefficients)
  colnames(s$coefficients)[1] <- "Odds Ratio"
  return(s)
}

check_assumptions <- function(model, data) {
  # Multicollinearity
  print(vif(model))
  
  # Linearity
  # Commented out, takes a really long time to runs
  
  # probabilities <- predict(model, type = "response")
  # predicted.classes <- ifelse(probabilities > 0.5, "helpful", "unhelpful")
  # num_data <- data[, colnames(data)[colnames(data) != 'bin_helpfulness']]
  # predictors <- colnames(num_data)
  # 
  # num_data <- num_data %>%
  #   mutate(logit = log(probabilities/(1-probabilities))) %>%
  #   gather(key = "predictors", value = "predictor.value", -logit)
  # ggplot(num_data, aes(logit, predictor.value)) +
  #   geom_point(size = 0.5, alpha = 0.5) +
  #   geom_smooth(method = "loess") +
  #   theme_bw() +
  #   facet_wrap(~predictors, scales = "free_y")

  # Influential Values
  plot(model, which = 4, id.n = 3)
}

evaluate_model <- function(model, test_data) {
  # Predict
  predictions <- predict(model, type='response', newdata = test_data)
  bin_predictions <- ifelse(predictions > 0.5, 1, 0)
  
  # Calculate F1 Score
  cf_matrix <- confusionMatrix(as.factor(bin_predictions),
                               as.factor(test_data$bin_helpfulness),
                               mode = 'everything', positive = '1')
  cat('\nConfusion Matrix')
  print(cf_matrix)
  
  cf_matrix_neg <- confusionMatrix(as.factor(bin_predictions),
                                   as.factor(test_data$bin_helpfulness),
                                   mode = 'everything', positive = '0')
  cat('\nConfusion Matrix Negative')
  print(cf_matrix_neg)
  
  # TDL
  decile_pred <- ntile(predictions, 10)
  
  deciles <- table(test_data$bin_helpfulness, 
                   decile_pred, dnn=c('Observed', 'Decile'))
  cat('\nTop Decile Table')
  print(deciles)
  
  cat('\nTDL:')
  tdl <- (deciles[2,10] / (deciles[1,10]+ deciles[2,10])) / mean(test_data$bin_helpfulness)
  print(tdl)
  
  # GINI
  #Make lift curve
  gini_predictions <- ROCR::prediction(predictions, test_data$bin_helpfulness)
  gini_performance <- performance(gini_predictions, 'tpr', 'fpr')
  plot(gini_performance, 
       xlab='Cumulative % of observations', 
       ylab='Cumulative % of positive cases', 
       xlim=c(0,1), ylim=c(0,1),xaxs='i',yaxs='i')
  abline(0,1, col='red')
  gini_auc <- performance(gini_predictions,'auc')
  #Calculate GINI
  gini <- as.numeric(gini_auc@y.values)*2-1
  cat('\nGINI: ')
  print(gini)
  cat('\n\n')
}

# Read and format data
train_data <- load_data('Video_Games_final_train.pkl.gz')
dev_data <- load_data('Video_Games_final_dev.pkl.gz')
test_data <- load_data('Video_Games_final_test.pkl.gz')

pdf(file='lr_plots.pdf')
sink(file='lr_output.txt')
# LR with only control variables
print('####LR CONTROL####')
lr_model_control <- glm(bin_helpfulness ~ rating_scl + review_time_scl + richness_scl, data=train_data, family='binomial')
summary_or(lr_model_control)
sign_coefs_control <- summary(lr_model_control)$coef[,4] < 0.05
only_sign_summ_control <- summary_or(lr_model_control)$coef[sign_coefs_control,]
stargazer(only_sign_summ_control)


check_assumptions(lr_model_control, train_data)

# print('##DEV RESULTS##')
# evaluate_model(lr_model_control, dev_data)

print('##TEST RESULTS##')
evaluate_model(lr_model_control, test_data)

# LR with only text variables
print('####LR TEXT####')
lr_model_text <- glm(bin_helpfulness ~ . -rating_scl-review_time_scl-richness_scl, data=train_data, family='binomial')
summary_or(lr_model_text)
sign_coefs_text <- summary(lr_model_text)$coef[,4] < 0.05
only_sign_summ_text <- summary_or(lr_model_text)$coef[sign_coefs_text,]
stargazer(only_sign_summ_text)

check_assumptions(lr_model_text, train_data)

# print('##DEV RESULTS##')
# evaluate_model(lr_model_text, dev_data)

print('##TEST RESULTS##')
evaluate_model(lr_model_text, test_data)

# LR with all variables
lr_model_full <- glm(bin_helpfulness ~ ., data=train_data, family='binomial')
summary_or(lr_model_full)
sign_coefs_full <- summary(lr_model_full)$coef[,4] < 0.05
only_sign_summ_full <- summary_or(lr_model_full)$coef[sign_coefs_full,]
stargazer(only_sign_summ_full)

check_assumptions(lr_model_full, train_data)

# print('##DEV RESULTS##')
# evaluate_model(lr_model_full, dev_data)

print('##TEST RESULTS##')
evaluate_model(lr_model_full, test_data)

sink(file=NULL)
dev.off()
