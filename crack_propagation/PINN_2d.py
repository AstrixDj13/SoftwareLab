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
    model.append(torch.nn.Linear(hidden_dim[-1], 3))
    # model.append(torch.nn.PReLU(num_parameters = 2, init = 0.25))
    
    return model

## Prediction of displacements U = [u,v]
def predictDisplacements(model, X, u0, domainLengthX, domainLengthY,s0):
   
    z = model(X)
    # different boundary conditions
    u = (X[:,0] - domainLengthX)*X[:,0]*z[:,0] + X[:,0]*u0/domainLengthX # u = 0 at x = 0 (left edge) and u = u0 at Lx (right edge)
    # u = ((X[:,0] - domainLengthX)*z[:,0] + u0/domainLengthX) * X[:,0]*X[:,1]/domainLengthY # u = 0 at both x = 0 and y = 0, and u varies linearly from 0 at y = 0 to u0 at y = Ly on right edge(x = Lx)  
    # u = X[:,0]*X[:,1]*z[:,0] # u = 0 at both x = 0 and y = 0 (left and bottom edge)

    # v = X[:,0]*X[:,1]*z[:,1] # v = 0 at both x = 0 and y = 0 (left and bottom edge)
    v = X[:,1]*z[:,1] # v = 0 only at y = 0 (bottom edge) 
    # v = z[:,1] # v free on both y-edges  
    
    s = s0 + z[:,2].view(-1,1)
    U = torch.cat((u.view(-1,1),v.view(-1,1)),1)

    return U, s

## Training
def trainModel(model, X, x, y, s0, u0, weights, jacobian, domainLengthX, domainLengthY, p, E, nu, eps, Gc, optimizer, epochs, val_X, val_x, val_y, val_s, val_weights, val_jacobian, analyticalSolution):
    # s = s0 # temporary requirement to prevent crack evolution
    # full batch GD
    for time in range(10):
        for i in range (epochs):
            U, s = predictDisplacements(model,X,u0,domainLengthX,domainLengthY,s0)

            #Total loss via costfunction
            loss = cost.costFunction(U, x,y, weights,jacobian, p,E,nu,s0,Gc,eps)
            
            # post processing data
            epochData.append(i+1)
            costData.append(loss.detach())
            if i%100 == 0:
                print(f"Epoch #{i+1}, loss = {loss}")
            trainingError.append(torch.abs(loss.detach()-analyticalSolution))    

            val_U, s_temp = predictDisplacements(model,val_X,u0,domainLengthX,domainLengthY,val_s)
            validationCost = cost.costFunction(val_U, val_x,val_y, val_weights, val_jacobian, p, E, nu,val_s,Gc,eps).detach()
            validationError.append(torch.abs(validationCost)-analyticalSolution)

            #Backpropagation
            torch.autograd.set_detect_anomaly(True)
            loss.backward(retain_graph = True)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
        s0 = s.detach() # required to start from a new compuational graph for each time step

    return U, s0, epochData, costData, trainingError, validationError 

epochData = []
costData = []
trainingError = [] 
validationError = [] 