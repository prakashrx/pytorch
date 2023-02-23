import torch
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


# 0) prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
print(f'X_numpy.shape = {X_numpy.shape}, y_numpy.shape = {y_numpy.shape}')

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) model
input_size = n_features
output_size = 1
model = torch.nn.Linear(input_size, output_size)

# 2) loss and optimizer
learning_rate = 0.01
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

X_test = torch.tensor([-3.0])
y_predicted = model(X_test)
print(f'prediction: f(3) = {y_predicted.item():.3f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()


# # 4) save the model
# torch.save(model.state_dict(), 'model.ckpt')

# # 5) load the model
# input_size = n_features
# output_size = 1
# model = torch.nn.Linear(input_size, output_size)
# model.load_state_dict(torch.load('model.ckpt'))
# model.eval()



