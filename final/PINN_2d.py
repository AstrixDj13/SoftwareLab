"""
Neural Network architecture and related functions
"""
import cost
import torch

## Definition of model
def buildModel(hidden_dim):
    # takes number of hidden layers as input and builds a NN model with 2 inputs and 3 outputs
    model = torch.nn.Sequential(torch.nn.Linear(2, hidden_dim[0]),torch.nn.Tanh()) 
    
    for i in range(len(hidden_dim)-1):
        model.append(torch.nn.Linear(hidden_dim[i],hidden_dim[i+1]))
        model.append(torch.nn.Tanh())
    model.append(torch.nn.Linear(hidden_dim[-1], 3))
    
    return model

## Setup the initial crack phase field
def initialCrack(graph, x0, x1, y0, y1):
    s = (graph[:, 0] > x0)
    s *= (graph[:, 0] < x1)
    s *= (graph[:, 1] > y0)
    s *= (graph[:, 1] < y1)
    return ~s.view(-1,1) + 0

## Prediction of displacements U = [u,v] and s-field
def predictDisplacements(model, X, u0, domainLengthX, domainLengthY, s0):
    # takes x and y as inputs, and returns u, v, ds and s as outputs
    z = model(X)

    ## Different Boundary Conditions
    u = X[:,0]*z[:,0] # u = 0 at x = 0 
    # u = (X[:,0] - domainLengthX)*X[:,0]*z[:,0] + X[:,0]*u0/domainLengthX # u = 0 at x = 0 (left edge) and u = u0 at Lx (right edge)
    # u = ((X[:,0] - domainLengthX)*z[:,0] + u0/domainLengthX) * X[:,0]*X[:,1]/domainLengthY # u = 0 at both x = 0 and y = 0, and u varies linearly from 0 at y = 0 to u0 at y = Ly on right edge(x = Lx)  
    # u = X[:,0]*X[:,1]*z[:,0] # u = 0 at both x = 0 and y = 0 (left and bottom edge)

    # v = z[:,1] # v free on both y-edges
    # v = X[:,0]*X[:,1]*z[:,1] # v = 0 at both x = 0 and y = 0 (left and bottom edge)
    # v = X[:,1]*z[:,1] # v = 0 only at y = 0 (bottom edge) 
    v = (X[:,0] - domainLengthX)*X[:,0]*z[:,1] - X[:,0]*u0/domainLengthX # shear

    U = torch.cat((u.view(-1,1),v.view(-1,1)), 1)
    ds = torch.nn.Sigmoid()(z[:,2].view(-1,1)) # restricts ds to positive values
    s = s0 - ds

    return U, s, ds

## Training
def trainModel(model, X, x, y, s0, u0, weights, jacobian, domainLengthX, domainLengthY, 
                p, E, nu, eps, Gc, optimizer, epochs, val_X, val_x, val_y, val_s, val_weights, val_jacobian, analyticalSolution,ts):
    
    # s = s0 # temporary requirement to prevent crack evolution
    # full batch Gadient Descent
    for time in range(ts):
        # loop over pseudo time-steps
        for i in range (epochs):
            # loop over epochs
            U, s, ds = predictDisplacements(model, X, u0, domainLengthX, domainLengthY, s0)

            # Total loss and energies from costfunction
            loss, elasticEnergy, crackEnergy = cost.costFunction(U, x, y, weights, jacobian, p, E, nu, s, Gc, eps, ds)
            
            # data processing for visualization
            epochData.append(i+1)
            costData.append(loss.detach())
            if i%100 == 0:
                print(f"Epoch #{i+1}, loss = {loss}")
            trainingError.append(torch.abs(loss.detach()-analyticalSolution))    
            elasticEnergyData.append(elasticEnergy.detach())
            crackEnergyData.append(crackEnergy.detach())

            #val_U, s_temp = predictDisplacements(model,val_X,u0,domainLengthX,domainLengthY,val_s)
            #validationCost = cost.costFunction(val_U, val_x,val_y, val_weights, val_jacobian, p, E, nu,val_s,Gc,eps).detach()
            #validationError.append(torch.abs(validationCost)-analyticalSolution)
            validationError=[]
            
            # Backpropagation
            loss.backward(retain_graph = True)
            optimizer.step()
            # reset all gradients to zero
            model.zero_grad()
            optimizer.zero_grad()

        s0 = s.detach()  # required to start from a new compuational graph for each time step
        # u0 = u0 + 0.1
        # s0[s0<0] = 0
        # s0[s0>1] = 1 
    return U, s0, epochData, costData, trainingError, validationError, elasticEnergyData, crackEnergyData

# empty lists for storing results after training
epochData = []
costData = []
trainingError = [] 
validationError = [] 
elasticEnergyData = []
crackEnergyData = []