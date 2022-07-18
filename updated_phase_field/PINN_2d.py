# -*- coding: utf-8 -*-
"""
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
    model.append(torch.nn.Linear(hidden_dim[-1], 2))
    
    return model

## Prediction of displacements U = [u,v]
def predictDisplacements(model, X, u0, domainLengthX, domainLengthY,s0):
    z = model(X)
    # different boundary conditions
    u = (X[:,0] - domainLengthX)*X[:,0]*z[:,0] + X[:,0]*u0/domainLengthX # u = 0 at x = 0 (left edge) and u = u0 at Lx (right edge)
    # u = ((X[:,0] - domainLengthX)*z[:,0] + u0/domainLengthX) * X[:,0]*X[:,1]/domainLengthY # u = 0 at both x = 0 and y = 0, and u varies linearly from 0 at y = 0 to u0 at y = Ly on right edge(x = Lx)  
    # u = X[:,0]*X[:,1]*z[:,0] # u = 0 at both x = 0 and y = 0 (left and bottom edge)

    # v = X[:,0]*X[:,1]*z[:,1] # v = 0 at both x = 0 and y = 0 (left and bottom edge)
    # v = X[:,1]*z[:,1] # v = 0 only at y = 0 (bottom edge) 
    v = z[:,1] # v free on both y-edges  

    # s = s0 + z[:,2].view(-1,1)
    U = torch.cat((u.view(-1,1),v.view(-1,1)),1)
    return U

## Training
def trainModel(model, X, x, y, s0, u0, weights, jacobian, domainLengthX, domainLengthY, p, E, nu, eps, Gc, optimizer, epochs, val_X, val_x, val_y, val_s, val_weights, val_jacobian, analyticalSolution):
    s = s0 # temporary requirement to prevent crack evolution
    # full batch GD
    for i in range (epochs):
        U = predictDisplacements(model,X,u0,domainLengthX,domainLengthY,s0)
        #Total loss via costfunction
        loss = cost.costFunction(U, x,y, weights,jacobian, p,E,nu,s,Gc,eps)
        
        # post processing data
        epochData.append(i+1)
        costData.append(loss.detach())
        if i%100 == 0:
            print(f"Epoch #{i+1}, loss = {loss.detach()}")
        trainingError.append(torch.abs(loss.detach()-analyticalSolution))    
        val_U = predictDisplacements(model,val_X,u0,domainLengthX,domainLengthY,s0)
        validationCost = cost.costFunction(val_U, val_x,val_y, val_weights, val_jacobian, p, E, nu,val_s,Gc,eps)
        validationError.append(torch.abs(validationCost-analyticalSolution))

        #Backpropagation
        loss.backward(retain_graph = True)
        optimizer.step()
        model.zero_grad()
        optimizer.zero_grad()
    
    return U, s, epochData, costData, trainingError, validationError 

epochData = []
costData = []
trainingError = [] 
validationError = [] 