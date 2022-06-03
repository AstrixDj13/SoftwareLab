# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:18:31 2022

@author: lukas
"""
import torch
import numpy as np

## Definition of model
def buildModel(hidden_dim):
    model = torch.nn.Sequential(torch.nn.Linear(2, hidden_dim[0]),torch.nn.Tanh()) 
    
    for i in range(len(hidden_dim)):
        model.append(torch.nn.Linear(hidden_dim[i],hidden_dim[i+1]),torch.nnTanh())
    model.append(torch.nn.Linear(hidden_dim[-1], 2))
    
    return model

## Prediction of displacement u = [u,v]
def predict_u(model, x, y, u_0):
    
    x = torch.reshape(x,(x.length(),1))
    y = torch.reshape(y,(y.length(),1))

    points = torch.cat((x,y), 1)
    
    u = torch.zeros_like(points)
    
    domainLength = x[-1];
    z = model(points)
    u[0,:] = (x-domainLength)*x*z[0,:] + x/domainLength*u_0
    u[1,:] = z[1,:]
    return u



#Setup size of hidden layers
size_hidden_dim = [10, 20, 10]

#Initialize model
model = buildModel(size_hidden_dim)

#Setup optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

#Training loop
epochs = 1000
for i in range (epochs):
    
    #Total loss via costfunction
    loss = costFunction()
    
    #Backpropagation
    loss.backward()
    optimizer.step()
    model.zero_grad()
    optimizer.zero_grad()