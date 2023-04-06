library(tidyverse)
library(kernlab)


df_embeddings <- unnest_wider(claims_embeddings, col = embeddings, names_sep = "_") |> 
  select(-1, -2)


# Split into train and test

set.seed(123)
train_index <- sample(nrow(df_embeddings), 0.7 * nrow(df_embeddings))
train_set <- df_embeddings[train_index, ]
test_set <- df_embeddings[-train_index, ]


# train model

svm_model <- ksvm(class ~ ., data = train_set, kernel = "vanilladot")
svm_predictions <- predict(svm_model, newdata = test_set)


# Evaluate the model

accuracy <- sum(svm_predictions == test_set$class) / nrow(test_set)
cat("The accuracy of the SVM model is:", accuracy, "\n")

caret::confusionMatrix(test_set$class, svm_predictions, mode =  'prec_recall')
