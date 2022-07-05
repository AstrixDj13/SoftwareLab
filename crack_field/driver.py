import gaussian
import cost
import plot_data
import PINN_2d
import torch
import matplotlib.pyplot as plt

#Setup size of hidden layers
size_hidden_dim = [10, 20, 10]

#Initialize model
model = PINN_2d.buildModel(size_hidden_dim)
print(model)
#Setup optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

# Mathematical modelling data 
domainLengthX = 1
domainLengthY = 1
numNodesX = 20
numNodesY = 20
numGPX = 2
numGPY = 2
x = torch.linspace(0, domainLengthX, numNodesX)
y = torch.linspace(0, domainLengthY, numNodesY)
X, weights, jacobian = gaussian.getGlobalMapping(x,y,numGPX,numGPY)

x = X[:,0].view(-1,1)
y = X[:,1].view(-1,1)
# setting x and y as leafs for graph computations
x.requires_grad = True
y.requires_grad = True
X = torch.cat((x,y),1)
# Problem data
E = lambda x,y: 1 + 0 * x + 0 * y
p = lambda x,y: 0#x.y
nu = lambda x,y: 0.4 
u_0 = 1
eps = 0.05
Gc = 0.5
#initial field
s0 = 1-torch.exp(-((x - 0.5*domainLengthX)**2 + (y - domainLengthY)**2)/eps**2)
#Training loop
epochs = 300


U, s, epochData, costData = PINN_2d.trainModel(model, X, x, y, u_0, weights, jacobian, domainLengthX, domainLengthY, p, E, nu, eps,Gc, optimizer, epochs)

## Surface plots
# reshaping vectors in grid form for 2d plots
x = x.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
y = y.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
u = U[:,0].reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
v = U[:,1].reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
s = s.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))

#plot initial field
s0 = s0.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
cb = plt.pcolormesh(x.detach(), y.detach(), s0.detach(), cmap='plasma', antialiased=False)
plt.colorbar(cb)
plt.show()

print(costData[-1])
print(f"s:{s}")
print(f"s0:{s0}")

plot_data.plotData(x,y,u,v,s,epochData,costData)

