import gaussian
import cost
import plot_data
import PINN_2d
import torch
import matplotlib.pyplot as plt

def initialCrack(graph, x0, x1, y0, y1):
	s = (graph[:, 0] > x0)
	s *= (graph[:, 0] < x1)
	s *= (graph[:, 1] > y0)
	s *= (graph[:, 1] < y1)
	return ~s+0

#Setup size of hidden layers
size_hidden_dim = [100, 200, 100]

#Initialize model
model = PINN_2d.buildModel(size_hidden_dim)
print(model)
#Setup optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = 0.002)

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
u0 = 1
eps = 0.1
Gc = 500
## Initial crack field
# continuous
s0 = 1-torch.exp(-((x - 0.5*domainLengthX)**2 + (y - 0.5*domainLengthY)**2)/eps**2) # point crack
# s0 = 1-torch.exp(-(torch.abs(x-0.5*domainLengthX)/eps)) # line crack
# discontinuous
s0 = initialCrack(X, 0.25,0.75,0.25,0.75)

# u = x*y*0 + u0
# v = x*y*0
# U = torch.cat((u,v),1)

# newcost = cost.getElasticEnergy(U,x,y,s0,weights, jacobian, E, nu, Gc, eps)
#Training loop
epochs = 200

U, s, epochData, costData = PINN_2d.trainModel(model, X, x, y, s0, u0, weights, jacobian, domainLengthX, domainLengthY, p, E, nu, eps,Gc, optimizer, epochs)
U = PINN_2d.predictDisplacements(model, X, u0, domainLengthX, domainLengthY,s0)

# ee, I1 = cost.getElasticEnergy(U, x,y, s, weights, jacobian, E, nu, Gc, eps)
# ## Validation
# # generate a new validation grid
# x = torch.linspace(0, domainLengthX, 10)
# y = torch.linspace(0, domainLengthY, 10)
# x,y = torch.meshgrid(x,y)
# X = torch.cat((x.reshape(-1,1),y.reshape(-1,1)),1)
# z = model(X)
# u = (X[:,0] - domainLengthX)*X[:,0]*z[:,0] + X[:,0]*u0/domainLengthX # u = 0 at x = 0 (left edge) and u = u0 at Lx (right edge)
# v = z[:,1] # free
# x = x.reshape(10,10)
# y = y.reshape(10,10)
# u = u.reshape(10,10)
# v = v.reshape(10,10)
# s = torch.zeros(10,10)

## Surface plots
# reshaping vectors in grid form for 2d plots
x = x.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
y = y.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
u = U[:,0].reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
v = U[:,1].reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
s = s.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))

# plot initial field
s0 = s0.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
cb = plt.pcolormesh(x.detach(), y.detach(), s0.detach(), cmap='plasma', antialiased=False)
plt.title("Initial Phase Field")
plt.colorbar(cb)
plt.show()

# I1 = I1.reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
# cb = plt.pcolormesh(x.detach(), y.detach(), I1.detach(), cmap='plasma', antialiased=False)
# plt.colorbar(cb)
# plt.show()

plot_data.plotData(x,y,u,v,s,epochData,costData)
print(f"Final cost:{costData[-1]}")
 

# print(f"s:{s}")
# print(f"s0:{s0}")

# # test for function to apply initial crack 
# c = initialCrack(X, 0.48,0.53,0.5,1).reshape(numGPX*(numNodesX-1), numGPY*(numNodesY-1))
# cb = plt.pcolormesh(x.detach(), y.detach(), c.detach(), cmap='plasma', antialiased=False)
# plt.colorbar(cb)
# plt.show()


