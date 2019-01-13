# R interface to Keras, using MNIST Example
# https://keras.rstudio.com/#tutorials

# devtools::install_github("rstudio/keras")
# install.packages("keras")
library(keras)
install_keras(tensorflow = "1.5")
# install_keras()
# install_keras(method = c("auto", "virtualenv", "conda"), conda = "auto", version = "default", tensorflow = "default", extra_packages = c("tensorflow-hub"))


# Preparing the Data #

# The MNIST dataset is included with Keras and can be accessed using the dataset_mnist() function. Here we load the dataset then create variables for our test and training data:
library(keras)
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

#The x data is a 3-d array (images,width,height) of grayscale values . To prepare the data for training we convert the 3-d arrays into matrices by reshaping width and height into a single dimension (28x28 images are flattened into length 784 vectors). Then, we convert the grayscale values from integers ranging between 0 to 255 into floating point values ranging between 0 and 1:
# reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))
# rescale
x_train <- x_train / 255
x_test <- x_test / 255

y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)


## Defining the Model ##
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

# Training and Evaluation

# Use the fit() function to train the model for 30 epochs using batches of 128 images:

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)

# The history object returned by fit() includes loss and accuracy metrics which we can plot:
plot(history)


# Evaluate the model’s performance on the test data:
model %>% evaluate(x_test, y_test)

# Generate predictions on new data:
model %>% predict_classes(x_test)



# Tutorials
# 
# To learn the basics of Keras, we recommend the following sequence of tutorials:
# 
#     Basic Classification — In this tutorial, we train a neural network model to classify images of clothing, like sneakers and shirts.
# 
#     Text Classification — This tutorial classifies movie reviews as positive or negative using the text of the review.
# 
#     Basic Regression — This tutorial builds a model to predict the median price of homes in a Boston suburb during the mid-1970s.
# 
#     Overfitting and Underfitting — In this tutorial, we explore two common regularization techniques (weight regularization and dropout) and use them to improve our movie review classification results.
# 
#     Save and Restore Models — This tutorial demonstrates various ways to save and share models (after as well as during training).
# 
# These tutorials walk you through the main components of the Keras library and demonstrate the core workflows used for training and improving the performance of neural networks. The Guide to Keras Basics provides a more condensed summary of this material.
# 
# The Deep Learning with Keras cheat sheet also provides a condensed high level guide to using Keras.
# Learning More
# 
# To learn more about Keras, you can check out these articles:
# 
#     Guide to the Sequential Model
# 
#     Guide to the Functional API
# 
#     Frequently Asked Questions
# 
#     Training Visualization
# 
#     Using Pre-Trained Models
# 
#     Keras with Eager Execution
# 
# The examples demonstrate more advanced models including transfer learning, variational auto-encoding, question-answering with memory networks, text generation with stacked LSTMs, etc.
# 
# The function reference includes detailed information on all of the functions available in the package.
# 
# Deep Learning with R Book
# 
# If you want a more comprehensive introduction to both Keras and the concepts and practice of deep learning, we recommend the Deep Learning with R book from Manning. This book is a collaboration between François Chollet, the creator of Keras, and J.J. Allaire, who wrote the R interface to Keras.
# 
# The book presumes no significant knowledge of machine learning and deep learning, and goes all the way from basic theory to advanced practical applications, all using the R interface to Kera