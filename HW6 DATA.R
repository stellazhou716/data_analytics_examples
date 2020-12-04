
library(tree)
library(randomForest)
library(haven)
library(glmnet) 
library(tidyverse)
library(plyr)
library(dplyr)
library(e1071)

test1 <- read_dta(file = "ca_school_testscore_1.dta")

test2 <- read_dta(file = "ca_school_testscore_2.dta")

# generate training and test dataset 

set.seed(123456)
n <- floor(0.75 * nrow(test1))
train_ind <- sample(seq_len(nrow(test1)), size = n)

train_1 <- test1[train_ind, ]
test_1 <- test1[-train_ind, ]

n <- floor(0.75 * nrow(test2))
train_ind <- sample(seq_len(nrow(test2)), size = n)

train_2 <- test2[train_ind, ]
test_2 <- test2[-train_ind, ]

#OLS using the SMALL dataset (k=4) 
fit1_train <- lm( testscore ~ te_avgyr_s + str_s + med_income_z + exp_1000_1999_d, data=train_2)
MSPE1_in_sample <- mean( (train_2$testscore - predict.lm(fit1_train, train_2))^2 )

fit1_test <- lm( testscore ~ te_avgyr_s + str_s + med_income_z + exp_1000_1999_d, data=test_2)
MSPE1_out_sample <- mean( (test_2$testscore - predict.lm(fit1_test, test_2))^2 )



#OLS using the LARGE dataset (k=817) 

fit2_train <- lm( testscore ~ ., data=train_1)
MSPE2_in_sample <- mean( (train_1$testscore - predict.lm(fit2_train, train_1))^2 )

fit2_test <- lm( testscore ~ ., data=test_1)
MSPE2_out_sample <- mean( (test_1$testscore - predict.lm(fit2_test, test_1))^2 )



#LASSO using the LARGE dataset (k=817) 


y <- train_1$testscore
x <- train_1 %>% dplyr::select(v1:cu_38) %>% data.matrix()

lambdas <- 10^seq(3, -2, by = -.1) 
fit3_train <- glmnet(x, y, alpha = 1, lambda = lambdas)
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = lambdas)
opt_lambda <- cv_fit$lambda.min
MSPE3_in_sample <- mean( (train_1$testscore - predict(fit3_train, s = opt_lambda, newx = x))^2 )
  
y <- test_1$testscore
x <- test_1 %>% dplyr::select(v1:cu_38) %>% data.matrix()

lambdas <- 10^seq(3, -2, by = -.1) 
fit3_test <- glmnet(x, y, alpha = 1, lambda = lambdas)
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = lambdas)
opt_lambda <- cv_fit$lambda.min
MSPE3_out_sample <- mean( (test_1$testscore - predict(fit3_test, s = opt_lambda, newx = x))^2 )



#Ridge using the LARGE dataset (k=817) 

y <- train_1$testscore
x <- train_1 %>% dplyr::select(v1:cu_38) %>% data.matrix()

lambdas <- 10^seq(3, -2, by = -.1) 
fit4_train <- glmnet(x, y, alpha = 0, lambda = lambdas)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas)
opt_lambda <- cv_fit$lambda.min
MSPE4_in_sample <- mean( (train_1$testscore - predict(fit4_train, s = opt_lambda, newx = x))^2 )

y <- test_1$testscore
x <- test_1 %>% dplyr::select(v1:cu_38) %>% data.matrix()

lambdas <- 10^seq(3, -2, by = -.1) 
fit4_test <- glmnet(x, y, alpha = 0, lambda = lambdas)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas)
opt_lambda <- cv_fit$lambda.min
MSPE4_out_sample <- mean( (test_1$testscore - predict(fit4_test, s = opt_lambda, newx = x))^2 )



                         
#Random Forest using the Medium dataset (k=38) 
fit5_train <- randomForest(testscore ~ ., data=train_2, ntree=1000,
                          keep.forest=TRUE, importance=TRUE)
MSPE5_in_sample <- mean( (train_2$testscore - predict(fit5_train, train_2))^2 )

fit5_test <- randomForest(testscore ~ ., data=test_2, ntree=1000,
                           keep.forest=TRUE, importance=TRUE)
MSPE5_out_sample <- mean( (test_2$testscore - predict(fit5_test, test_2))^2 )


d <- data.frame("OLS (k=4)"=c(MSPE1_in_sample,MSPE1_out_sample),
                "OLS (k=817)"=c(MSPE2_in_sample,MSPE2_out_sample),
                "LASSO (k=817)"=c(MSPE3_in_sample,MSPE3_out_sample),
                "Ridge (k=817)"=c(MSPE4_in_sample,MSPE4_out_sample),
                "Random Forest (k=38)"=c(MSPE5_in_sample,MSPE5_out_sample))
rownames(d) <- c(" In-sample root MSPE ", " Out-of-sample root MSPE ")

d

