# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:25:21 2022

@author: lukas
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

def buildDomain(x_begin, x_end, numberIntervals):
    
    xi_GP = torch.zeros(2)

    xi_GP[0] = -np.sqrt(1/3)
    xi_GP[1] = np.sqrt(1/3)
    
    x = torch.zeros(numberIntervals*2)
    
    dx = (x_end-x_begin)/(numberIntervals)
    
    N1 = lambda xi: (xi-1)/(-2)
    N2 = lambda xi: (xi+1)/2
    
    det = torch.zeros(numberIntervals)
    
    for i in range(numberIntervals):
        start = i*dx
        end = (i+1)*dx
        x[2*i]   = N1(xi_GP[0])*start + N2(xi_GP[0])*end
        x[2*i+1] = N1(xi_GP[1])*start + N2(xi_GP[1])*end
        det[i] = -0.5*start + 0.5*end

    return [x, det]
    
def buildModel(input_dim, hidden_dim, output_dim):
    model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
    torch.nn.Tanh(), 
    torch.nn.Linear(hidden_dim, output_dim))
    return model

def get_derivative(y, x):
    dydx = grad(y, x, torch.ones(x.size()[0],1), create_graph = True, retain_graph = True)[0]
    return dydx

def predict_u(model, x):
    z = model(x)
    u = (x-1)*x*z
    return u

def gaussian_integration(u, numberIntervals, det):
    value = 0.0

    for i in range(numberIntervals):
        value += (u[2*i] + u[2*i+1]) * det[i]
        
    return value

def trapezoidal_integration(u, x):
    dx = x[1] - x[0]
    result = torch.sum(u)
    result = result - (u[0]+u[-1])/2
    return result*dx

def internal_energy(model, x, EA, numberIntervals, det):
    u = predict_u(model, x)
    du_dx = get_derivative(u, x)
    #internalEnergy = gaussian_integration(0.5*EA*du_dx**2, numberIntervals, det)
    internalEnergy = trapezoidal_integration(EA*du_dx**2, x)
    return internalEnergy

def external_energy(model, x, p, numberIntervals, det):
    integrant = -p(x)*predict_u(model, x)
    #externalEnergy = gaussian_integration(integrant, numberIntervals, det)
    externalEnergy = trapezoidal_integration(integrant, x)
    return externalEnergy
    
#Initializing Model
model = buildModel(1, 10, 1)

#Setting-up optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

#Initializing parameters
EA = 1.0
p = lambda x: 4*np.pi**2*torch.sin(2*np.pi*x)

#Initializing domain
xbegin = 0
xend   = 1
numberIntervals = 20

[x,det] = buildDomain(xbegin, xend, numberIntervals)
x = x.view(-1, 1)
x.requires_grad_(requires_grad = True)
x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)
det = torch.zeros(10)

### Begin training-loop
epochs = 1000

costs = np.zeros(epochs)

for i in range (epochs):
    
    #Total loss: cost function = total potential energy
    loss = internal_energy(model, x, EA, numberIntervals, det) + external_energy(model, x, p, numberIntervals, det)
    costs[i] = loss
    
    #Backpropagation
    loss.backward()
    optimizer.step()
    model.zero_grad()
    optimizer.zero_grad()
    
#analytical solution
u = lambda x: np.sin(2*np.pi*x)
#plt.plot(costs)
plt.plot(x.detach(),predict_u(model,x).detach(),'ro')
plt.plot(x.detach(), u(x.detach()))




