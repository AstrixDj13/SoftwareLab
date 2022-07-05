# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:18:31 2022

@author: lukas
"""
import cost
import torch

## Definition of model
def buildModel(hidden_dim):
    model = torch.nn.Sequential(torch.nn.Linear(2, hidden_dim[0]),torch.nn.Tanh()) 
    
    for i in range(len(hidden_dim)-1):
        model.append(torch.nn.Linear(hidden_dim[i],hidden_dim[i+1]))
        model.append(torch.nn.Tanh())
    model.append(torch.nn.Linear(hidden_dim[-1], 3))
    
    return model

## Prediction of displacements U = [u,v]
def predictDisplacements(model, X, u_0, domainLengthX, domainLengthY,s0):
    z = model(X)
    # different boundary conditions
    u = (X[:,0] - domainLengthX)*X[:,0]*z[:,0] + X[:,0]*u_0/domainLengthX # u = 0 at x = 0 (left edge) and u = u0 at Lx (right edge)
    # u = ((X[:,0] - domainLengthX)*z[:,0] + u_0/domainLengthX) * X[:,0]*X[:,1]/domainLengthY # u = 0 at both x = 0 and y = 0, and u varies linearly from 0 at y = 0 to u0 at y = Ly on right edge(x = Lx)  
    # u = X[:,0]*X[:,1]*z[:,0] # u = 0 at both x = 0 and y = 0 (left and bottom edge)
    
    s = s0.view(1,-1) + z[:,2]

    # v = X[:,0]*X[:,1]*z[:,1] # v = 0 at both x = 0 and y = 0 (left and bottom edge)
    # v = X[:,1]*z[:,1] # v = 0 only at y = 0 (bottom edge) 
    v = z[:,1] # v free on both y-edges  
    U = torch.cat((u.view(-1,1),v.view(-1,1)),1)
    
    return U, s

## Training
def trainModel(model, X, x, y, u_0, weights, jacobian, domainLengthX, domainLengthY, p, E, nu, eps, Gc, optimizer, epochs):
    # full batch GD
    for i in range (epochs):
        s0 = 1-torch.exp(-((x - 0.5*domainLengthX)**2 + (y - domainLengthY)**2)/eps**2)
        U,s = predictDisplacements(model,X,u_0,domainLengthX,domainLengthY,s0)

        #Total loss via costfunction
        loss = cost.costFunction(U, x,y, weights,jacobian, p,E,nu,s,Gc,eps)
        # post processing data
        epochData.append(i+1)
        costData.append(loss.detach())

        #Backpropagation
        loss.backward()
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()
    
    return U, s, epochData, costData 

epochData = []
costData = []