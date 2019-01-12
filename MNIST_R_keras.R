
# https://keras.rstudio.com/#tutorials

# install.packages("keras")
library(keras)
install_keras()
install_keras(method = c("auto", "virtualenv", "conda"),
  conda = "auto", version = "default", tensorflow = "default",
  extra_packages = c("tensorflow-hub"))



library(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y