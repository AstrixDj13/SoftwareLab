"""
title: PINN for linear static bar under sine loading
author: Hritik Singh
"""

import torch
import torch.autograd as ag
import math
import matplotlib.pyplot as plt

def buildModel(input_dim, hidden_dim, output_dim):
	model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
	torch.nn.Tanh(),
	torch.nn.Linear(hidden_dim, output_dim))
	return model

def get_derivative(y, x):
	dydx = ag.grad(y, x, torch.ones(x.size()[0], 1),
	create_graph=True,
	retain_graph=True)[0]
	return dydx

def f(model, x, EA, p):
	u = model(x)
	u_x = get_derivative(u, x)
	EAu_xx = get_derivative(EA(x) * u_x, x)
	f = EAu_xx + p(x)
	return f
	
# Training data
N = 10
x = torch.linspace(0, 1, N, requires_grad=True).view(-1, 1) # N points b/w l = (0,1) 

# Model initialization
model = buildModel(1, 10, 1)
EA = lambda x: 1 + 0 * x # constant EA
p = lambda x: 4 * math.pi**2 * torch.sin(2 * math.pi * x) # applied distributed load
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

# Dirichlet boundary conditions
u0 = 0
u1 = 0

costs = []
ep = []

# PINN training
epochs = 500
for epoch in range(epochs):
	
	fn = f(model, x, EA, p)
	MSE_f = torch.sum(fn**2)
	
	u0_pred = model(torch.tensor([0.0]))
	u1_pred = model(torch.tensor([1.0]))
	MSE_b = (u0_pred - u0)**2 + (u1_pred - u1)**2

	cost_fn = 100*MSE_b + MSE_f
	costs.append(cost_fn)
	ep.append(epoch+1)
	
	# Backpropagation
	optimizer.zero_grad()
	cost_fn.backward()
	optimizer.step()

# Validation
n = 20
X = torch.linspace(0, 1, n, requires_grad=True).view(-1, 1) # n points along bar for validation
u = torch.sin(2 * math.pi * X) # analytical/actual displacements
u_pred = model(X) # displacements predicted by PINN

# Plots
fig, ax = plt.subplots(1,2)
ax[0].set_xlabel("x")
ax[0].set_ylabel("Displacement, u(x)")
ax[0].plot(X.detach(),u.detach(),'r.',label='Analytical')
ax[0].plot(X.detach(),u_pred.detach(),'b.',label='Prediction')
ax[0].legend()
ax[1].set_ylabel("Cost function")
ax[1].set_xlabel("Epochs")
ax[1].plot(ep,costs,'g.')
plt.show()
