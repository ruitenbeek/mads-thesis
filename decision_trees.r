# install.packages('reticulate')
library(reticulate)
library(dplyr)
library(tidyr)
library(caret)
library(ROCR)
#install.packages(rpart.plot)
library(rpart.plot)
library(rpart)
library(partykit)
pd <- import("pandas")

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

evaluate_model <- function(model, test_data) {
  # Predict
  predictions <- predict(model, newdata=test_data, type ="prob")[,2]
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

pdf(file='df_plots.pdf', width=35, height=30)
sink(file='dt_output.txt')
# CART tree with only control variables
print('####CART TREE CONTROL####')
settings_control <- rpart.control(minsplit = 1000, cp = 0.001, maxdepth = 30)
cart_tree_control <- rpart(bin_helpfulness ~ rating_scl+review_time_scl+richness_scl, 
                           data=train_data, method='class', control=settings_control)
cart_tree_control_visual <- as.party(cart_tree_control)
plot(cart_tree_control_visual, type='simple')

print('##DEV RESULTS##')
evaluate_model(cart_tree_control, dev_data)

print('##TEST RESULTS##')
evaluate_model(cart_tree_control, test_data)


# CART tree with only text variables
print('####CART TREE TEXT####')
settings_text <- rpart.control(minsplit = 500, cp = 0.001, maxdepth = 15)
cart_tree_text <- rpart(bin_helpfulness ~ .-rating_scl-review_time_scl-richness_scl, 
                           data=train_data, method='class', control=settings_text)
cart_tree_text_visual <- as.party(cart_tree_text)
plot(cart_tree_text_visual, type='simple')

print('##DEV RESULTS##')
evaluate_model(cart_tree_text, dev_data)

  print('##TEST RESULTS##')
evaluate_model(cart_tree_text, test_data)

# CART tree with all variables
print('####CART TREE FULL####')
settings_full <- rpart.control(minsplit = 500, cp = 0.001, maxdepth = 15)
cart_tree_full <- rpart(bin_helpfulness ~ ., 
                        data=train_data, method='class', control=settings_full)
cart_tree_full_visual <- as.party(cart_tree_full)
plot(cart_tree_full_visual, type='simple')

print('##DEV RESULTS##')
evaluate_model(cart_tree_full, dev_data)

print('##TEST RESULTS##')
evaluate_model(cart_tree_full, test_data)

sink(file=NULL)
dev.off()
rpart.plot(cart_tree_control, type=3, clip.right.labs=FALSE, branch=.3, fallen.leaves = FALSE)
rpart.plot(cart_tree_text, type=3, clip.right.labs=FALSE, branch=.9, fallen.leaves = FALSE, tweak=1.2, compress=FALSE, ycompress=FALSE)
rpart.plot(cart_tree_full, type=3, clip.right.labs=FALSE, branch=.9, fallen.leaves = FALSE, tweak=1.2, compress=FALSE, ycompress=FALSE)
