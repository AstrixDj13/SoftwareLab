import torch
import torch.autograd as ag
import math
import matplotlib.pyplot as plt

def buildModel(input_dim, hidden_dim, output_dim):
	model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim),
	torch.nn.Tanh(),
	torch.nn.Linear(hidden_dim, output_dim))
	return model

def get_derivative(y, x):
	dydx = ag.grad(y, x, torch.ones(x.size()[0], 1),
	create_graph=True,
	retain_graph=True)[0]
	return dydx

def f(model, x, EA, p):
	u = model(x)
	u_x = get_derivative(u, x)
	EAu_xx = get_derivative(EA(x) * u_x, x)
	f = EAu_xx + p(x)
	return f
	
# training data
N = 20
X = torch.linspace(0, 1, N, requires_grad=True).view(-1, 1)

model = buildModel(1, 10, 1)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

# Dirichlet boundary conditions
u0 = 0
u1 = 0

epochs = 5000
# Training 
for epoch in range(epochs):
	u_pred = model(X)
	
	x = torch.linspace(0, 1, 10, requires_grad=True).view(-1, 1) # 10 points b/w l = (0,1) 
	EA = lambda x: 1 + 0 * x # constant EA
	p = lambda x: 4 * math.pi**2 * torch.sin(2 * math.pi * x) # applied distributed load
	fn = f(model, x, EA, p)
	MSE_f = torch.sum(fn**2)
	
	#model = buildModel(1, 10, 1)
	u0_pred = model(torch.tensor([0.0]))
	u1_pred = model(torch.tensor([1.0]))
	MSE_b = (u0_pred - u0)**2 + (u1_pred - u1)**2

	cost_fn = MSE_b + MSE_f
	
	# Backpropagation
	optimizer.zero_grad()
	cost_fn.backward()
	optimizer.step()

plt.plot(X.detach(),u_pred.detach(),'ro')
plt.show()

