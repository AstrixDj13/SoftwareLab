# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 18:42:54 2022

@author: lukas
"""

import torch
from torch.autograd import grad
import math

def buildModel(input_dim, hidden_dim, output_dim):
    model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.Tanh(), torch.nn.Linear(hidden_dim, output_dim))
    return model

def get_derivative(y, x):
    dydx = grad(y, x, torch.ones(x.size()[0],1), create_graph = True, retain_graph = True)[0]
    return dydx

def f(model, x, EA, p):
    u = model(x)
    u_x = get_derivative(u, x)
    EAu_xx = get_derivative(EA(x)*u_x, x)
    f = EAu_xx + p(x)
    return f
    
#Initializing Model
model = buildModel(1, 10, 1)
x = torch.linspace(0, 1, 10, requires_grad = True).view(-1, 1)
EA = lambda x: 1 + 0*x
p = lambda x: 4*math.pi**2*torch.sin(2*math.pi*x)

#Setting up Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)

### Begin training-loop
epochs = 1000
for i in range (epochs):
    optimizer.zero_grad()
    
    #Function error
    lhs = f(model, x, EA, p)
    
    MSE_f = torch.sum(lhs**2)
    
    #Boundary error
    u0 = 0
    u1 = 0
    u0_pred = model(torch.tensor([0.]))
    u1_pred = model(torch.tensor([1.]))
    MSE_b = (u0-u0_pred)**2 + (u1-u1_pred)**2
    
    #Total Loss and Backpropagation
    loss = MSE_b + MSE_f
    loss.backward()
    optimizer.step()
    model.zero_grad()

print(model(torch.tensor([0.])))










