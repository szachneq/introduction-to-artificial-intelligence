# https://machinelearningmastery.com/develop-your-first-neural-network-with-pytorch-step-by-step

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

# create a tensor out of NumPy arrays
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(
    # The model expects rows of data with 8 variables (the first argument)
    # The first hidden layer has 12 neurons (the second argument),
    # followed by a ReLU activation function (rectified linear unit activation function)
    nn.Linear(8, 12),
    nn.ReLU(),
    # The second hidden layer has 8 neurons,
    # followed by another ReLU activation function
    nn.Linear(12, 8),
    nn.ReLU(),
    # The output layer has one neuron,
    # followed by a sigmoid activation function
    nn.Linear(8, 1),
    nn.Sigmoid()
)

# Training a network means finding the best set of weights to map inputs to outputs in your dataset.
# The loss function is the metric to measure the prediction’s distance to y.
# In this example, you should use binary cross entropy because it is a binary classification problem.

# Once you decide on the loss function, you also need an optimizer.
# The optimizer is the algorithm you use to adjust the model weights progressively to produce a better output.
# There are many optimizers to choose from, and in this example, Adam is used.
# This popular version of gradient descent can automatically tune itself and gives good results in a wide range of problems.

loss_fn = nn.BCELoss()  # binary cross entropy
# The optimizer usually has some configuration parameters.
# Most notably, the learning rate lr.
# But all optimizers need to know what to optimize.
# Therefore. you pass on model.parameters(), which is a generator of all parameters from the model you created.
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Epoch: Passes the entire training dataset to the model once
# Batch: One or more samples passed to the model, from which the gradient descent algorithm will be executed for one iteration

# Simply speaking, the entire dataset is split into batches,
#   and you pass the batches one by one into a model using a training loop. 
# Once you have exhausted all the batches, you have finished one epoch.
# Then you can start over again with the same dataset and start the second epoch,
#   continuing to refine the model.
# This process repeats until you are satisfied with the model’s output.

# IMPORTANT FOR REPORT:
# The size of a batch is limited by the system’s memory.
# Also, the number of computations required is linearly proportional to the size of a batch.
# The total number of batches over many epochs is how many times you run the gradient descent to refine the model.
# It is a trade-off that you want more iterations for the gradient descent so you can produce a better model,
#   but at the same time, you do not want the training to take too long to complete.
# The number of epochs and the size of a batch can be chosen experimentally by trial and error.

# The simplest way to build a training loop is to use two nested for-loops, one for epochs and one for batches:
n_epochs = 100 # wiecej epok zazwyczaj daje lepzse wyniki ale z czasem sa to diminishing returns
batch_size = 10 # mniejszy batch daje lepsze wyniki ale wolniej sie robi

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')
    
# W TYM PRZYKLADZIE JEST POMINIETE DZIELENIE DANYCH NA TRENINGOWE I TESTOWE!!!

# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)

accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# Ideally, you would like the loss to go to zero and the accuracy to go to 1.0 (e.g., 100%).
# This is not possible for any but the most trivial machine learning problems.
# Instead, you will always have some error in your model.
# The goal is to choose a model configuration and training configuration that achieves the lowest loss and highest accuracy possible for a given dataset.

# Neural networks are stochastic algorithms, meaning that the same algorithm on the same data can train a different model with different skill each time the code is run.

# ---

# You can adapt the above example and use it to generate predictions on the training dataset,
#   pretending it is a new dataset you have not seen before.

# make probability predictions with the model
predictions = model(X)
# print(predictions)
# round predictions
rounded = predictions.round() # to nearest 
# print(rounded)

# Alternately, you can convert the probability into 0 or 1 to predict crisp classes directly
# make class predictions with the model
predictions = (model(X) > 0.5).int()
# print(predictions)