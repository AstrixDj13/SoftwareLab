# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:18:31 2022

@author: lukas
"""
import gaussian
import cost
import torch
import numpy as np
import matplotlib.pyplot as plt

## Definition of model
def buildModel(hidden_dim):
    model = torch.nn.Sequential(torch.nn.Linear(2, hidden_dim[0]),torch.nn.Tanh()) 
    
    for i in range(len(hidden_dim)-1):
        model.append(torch.nn.Linear(hidden_dim[i],hidden_dim[i+1]))
        model.append(torch.nn.Tanh())
    model.append(torch.nn.Linear(hidden_dim[-1], 2))
    
    return model

## Prediction of displacement u = [u,v]
def predict_u(model, X, u_0):
    domainLength = 2; #change

    z = model(X)
    u = (X[:,0]-domainLength)*X[:,0]*torch.squeeze(z[:,0]) + X[:,0]*(u_0/domainLength)
    v = (X[:,1]-domainLength)*X[:,1]*torch.squeeze(z[:,1])
  
    U = torch.cat((u.view(-1,1),v.view(-1,1)),1)
    return U



#Setup size of hidden layers
size_hidden_dim = [10, 20, 10]

#Initialize model
model = buildModel(size_hidden_dim)
print(model)
#Setup optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

#Training loop
epochs = 100

u_0 = 1
x = torch.linspace(0, 2, 10)
y = torch.linspace(0, 2, 10)
[X, weights, jacobian] = gaussian.getGlobalMapping(x,y,2,2)

x = X[:,0].view(-1,1)
y = X[:,1].view(-1,1)
# setting x and y as leafs for graph computations
x.requires_grad = True
y.requires_grad = True
X = torch.cat((x,y),1)

E = lambda x,y: 1 + 0 * x + 0 * y
p = lambda x,y: 4 * y * torch.pi ** 2 * torch.sin(2 * torch.pi * x)
nu = lambda x,y: 0.5 + 0 * x + 0 * y

for i in range (epochs):
    U = predict_u(model, X,u_0)
    
    #Total loss via costfunction
    loss = cost.costFunction(U,x,y, weights,jacobian,p, E,nu)
    print(loss)
    #Backpropagation
    loss.backward()
    optimizer.step()
    #model.zero_grad()
    optimizer.zero_grad()
 # change to surface plots    
plt.plot(X[:,0].detach(), U[:,0].detach())
plt.show()