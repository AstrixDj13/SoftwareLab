import gaussian
import cost
import plot_data
import PINN_2d
import torch
import time
import matplotlib.pyplot as plt

# Checks if cuda is available or not, if available select it as device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Setup size of hidden layers
size_hidden_dim = [100, 200, 100]

# Initialize model
model = PINN_2d.buildModel(size_hidden_dim).to(device)
print(model)

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-3)

##--- Mathematical modelling of data like domain lengths, number of nodes and number of Gauss points ---##
domainLengthX = 1
domainLengthY = 1
numNodesX = 31
numNodesY = 31
numGPX = 4
numGPY = 4

# generate nodes along x and y directions
x = torch.linspace(0, domainLengthX, numNodesX)
y = torch.linspace(0, domainLengthY, numNodesY)

# Calculate Gauss points, corresponding weights and the Jacobian matrix
X, weights, jacobian = gaussian.getGlobalMapping(x, y, numGPX, numGPY)
weights = weights.to(device)
jacobian = jacobian.to(device)
x = X[:, 0].view(-1, 1).to(device)
y = X[:, 1].view(-1, 1).to(device)

# setting x and y as leafs for graph computations
x.requires_grad = True
y.requires_grad = True
X = torch.cat((x, y), 1).to(device)

##--- Problem data ---##
# Material parameters
E = lambda x, y: 100 + 0 * x + 0 * y
nu = lambda x, y: 0.4
# Loading parameters
p = lambda x, y: 0 # external load = 0 for our cases
u0 = 1 # initial displacement
# Crack parameters
eps = 0.03 # length parameter for continuous cracks
Gc = 7 # crack strength

##--- Initial crack field ---##
# continuous
# s0 = 1 - torch.exp(-((x - 0.5 * domainLengthX) ** 2 + (y - 0.5 * domainLengthY) ** 2) / eps ** 2)  # point crack
# s0 = 1-torch.exp(-(torch.abs(x-0.5*domainLengthX)/eps)) # line crack

# discontinuous
s0 = PINN_2d.initialCrack(X, 0.47*domainLengthX, 0.53*domainLengthX, 0.5*domainLengthY, domainLengthY) # vertical crack
# s0 = PINN_2d.initialCrack(X, 0.5*domainLengthX, domainLengthX, 0.47*domainLengthY, 0.53*domainLengthY) # horizontal crack 

##--- Validation ---##
analyticalSolution = 0
u_analytical = x>(0.5*domainLengthX)
v_analytical = torch.zeros(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))

# generate a new validation grid
val_numNodesX = 4
val_numNodesY = 4
val_x = torch.linspace(0, domainLengthX, val_numNodesX)
val_y = torch.linspace(0, domainLengthY, val_numNodesY)
val_X, val_weights, val_jacobian = gaussian.getGlobalMapping(val_x,val_y,4,4)
val_weights = val_weights.to(device)
val_jacobian = val_jacobian.to(device)
val_x = val_X[:,0].view(-1,1).to(device)
val_y = val_X[:,1].view(-1,1).to(device)
# setting validation x and y as leafs for graph computations
val_x.requires_grad = True
val_y.requires_grad = True
val_X = torch.cat((val_x,val_y),1).to(device)
val_s = 1-torch.exp(-(torch.abs(val_x-0.5*domainLengthX)/eps)) # change this to same as s0 when testing different configurations
# val_s = 1 - torch.exp(-((val_x - 0.5 * domainLengthX) ** 2 + (val_y - 0.5 * domainLengthY) ** 2) / eps ** 2) 
# val_s = initialCrack(val_X, 0.45*domainLengthX, 0.55*domainLengthX, 0.5*domainLengthY, domainLengthY) 
# val_s =  (val_y>=0.5)*(1 - torch.exp(-(torch.abs(val_x - 0.5 * domainLengthX) / eps))) + (val_y<0.5) 

##--- Training ---##
epochs = 10
ts = 1 # pseudo-time steps
tic = time.time() # start the stopwatch
U, s, epochData, costData, trainingError, validationError, elasticEnergy, crackEnergy = PINN_2d.trainModel(model, X, x, y, s0, u0, weights, jacobian, domainLengthX, domainLengthY,
                                               p, E, nu, eps, Gc, optimizer, epochs, val_X, val_x, val_y, val_s, val_weights, val_jacobian, analyticalSolution, ts)
toc = time.time() # stop the stopwatch
print(f"Final cost:{costData[-1]}")
print(f"Total time elapsed: {toc-tic} seconds")
print(f"Min. cost:{min(costData)}")

##--- Plots ---##
# reshaping vectors in grid form for 2d plots
x = x.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
y = y.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
u = U[:, 0].reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
v = U[:, 1].reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
s = s.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
u_analytical = u_analytical.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))
s0 = s0.reshape(numGPX * (numNodesX - 1), numGPY * (numNodesY - 1))

# plot crack fields
fig, ax = plt.subplots(1,2)
for col in range(2):
    ax[col].set_xlabel("x")
    ax[col].set_ylabel("y")

ax[0].set_title("Initial Phase Field")
cb = ax[0].pcolormesh(x.cpu().detach(), y.cpu().detach(), s0.cpu().detach(), cmap='plasma', antialiased=False)
ax[0].set_aspect('equal', adjustable='box')
fig.colorbar(cb, ax = ax[0])
ax[1].set_title("Updated Phase Field")
cb = ax[1].pcolormesh(x.cpu().detach(), y.cpu().detach(), s.cpu().detach(), cmap='plasma', antialiased=False)
ax[1].set_aspect('equal', adjustable='box')
fig.colorbar(cb, ax = ax[1])
plt.show()

# plot the displacement fields
plot_data.plotData(x.cpu(), y.cpu(), u.cpu(), v.cpu(), s.cpu(), u_analytical.cpu(), v_analytical.cpu(), epochData, costData, trainingError, validationError, elasticEnergy, crackEnergy)