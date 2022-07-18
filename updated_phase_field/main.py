import gaussian
import cost
import plot_data
import PINN_2d
import torch
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(f"Device: {device}")

def initialCrack(graph, x0, x1, y0, y1):
    s = (graph[:, 0] > x0)
    s *= (graph[:, 0] < x1)
    s *= (graph[:, 1] > y0)
    s *= (graph[:, 1] < y1)
    return ~s.view(-1,1) + 0

# Setup size of hidden layers
size_hidden_dim = [100, 200, 100]

# Initialize model
model = PINN_2d.buildModel(size_hidden_dim).to(device)
print(model)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

## Mathematical modelling data 
domainLengthX = 1
domainLengthY = 1
numNodesX = 20
numNodesY = 20
numGPX = 2
numGPY = 2
x = torch.linspace(0, domainLengthX, numNodesX)
y = torch.linspace(0, domainLengthY, numNodesY)
X, weights, jacobian = gaussian.getGlobalMapping(x, y, numGPX, numGPY)
weights = weights.to(device)
jacobian = jacobian.to(device)
x = X[:, 0].view(-1, 1).to(device)
y = X[:, 1].view(-1, 1).to(device)
# setting x and y as leafs for graph computations
x.requires_grad = True
y.requires_grad = True
X = torch.cat((x, y), 1).to(device)

# Problem data
E = lambda x, y: 1 + 0 * x + 0 * y
p = lambda x, y: 0  # x*y
nu = lambda x, y: 0.4
u0 = 1
eps = 0.05 # length parameter for continuous cracks
Gc = 500 # crack stiffness

## Initial crack field
# continuous
# s0 = 1 - torch.exp(-((x - 0.5 * domainLengthX) ** 2 + (y - 0.5 * domainLengthY) ** 2) / eps ** 2)  # point crack
s0 = 1-torch.exp(-(torch.abs(x-0.5*domainLengthX)/eps)) # line crack
# s0 = (y>=0.5)*(1 - torch.exp(-(torch.abs(x - 0.5 * domainLengthX) / eps))) + (y<0.5) # half-line crack in y direction
# s0 = (x>=0.5)*(1 - torch.exp(-(torch.abs(y - 0.5 * domainLengthY) / eps))) + (x<0.5) # half-line crack in x direction
# discontinuous
# s0 = initialCrack(X, 0.48*domainLengthX, 0.53*domainLengthX, 0.5*domainLengthY, domainLengthY) 

## Validation
analyticalSolution = 0
# generate a new validation grid
val_numNodesX = 10
val_numNodesY = 10
val_x = torch.linspace(0, domainLengthX, val_numNodesX)
val_y = torch.linspace(0, domainLengthY, val_numNodesY)
val_X, val_weights, val_jacobian = gaussian.getGlobalMapping(val_x,val_y,1,1)
val_weights = val_weights.to(device)
val_jacobian = val_jacobian.to(device)
val_x = val_X[:,0].view(-1,1).to(device)
val_y = val_X[:,1].view(-1,1).to(device)
# setting x and y as leafs for graph computations
val_x.requires_grad = True
val_y.requires_grad = True
val_X = torch.cat((val_x,val_y),1).to(device)
val_s = 1-torch.exp(-(torch.abs(val_x-0.5*domainLengthX)/eps)) # change this to same as s0 when testing different configurations

# Training loop
epochs = 1000
tic = time.time()
U, s, epochData, costData, trainingError, validationError = PINN_2d.trainModel(model, X, x, y, s0, u0, weights, jacobian, domainLengthX, domainLengthY,
                                               p, E, nu, eps, Gc, optimizer, epochs, val_X, val_x, val_y, val_s, val_weights, val_jacobian, analyticalSolution)
toc = time.time()
print(f"Final cost:{costData[-1]}")
print(f"Total time elapsed: {toc-tic} seconds")

## Surface plots
# reshaping vectors in grid form for 2d plots
x = x.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
y = y.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
u = U[:, 0].reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
v = U[:, 1].reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
s = s.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))

# plot initial field
s0 = s0.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
cb = plt.pcolormesh(x.cpu().detach(), y.cpu().detach(), s0.cpu().detach(), cmap='plasma', antialiased=False)
plt.title("Initial Phase Field")
plt.colorbar(cb)
plt.show()

plot_data.plotData(x.cpu(), y.cpu(), u.cpu(), v.cpu(), s.cpu(), epochData, costData, trainingError, validationError)
