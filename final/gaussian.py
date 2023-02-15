"""
Gaussian Mapping and Gaussian Integration upto 5 Gauss points
"""
import torch
import math

def getGaussianVariables(numGP):
	""" Calculates the position of the gauss points and their corresponding weights
		based on the number of gauss points given as parameter
		Can handle upto 5 gauss points """
	if numGP == 1:
		xi_GP = torch.tensor([0.])
		weights = torch.tensor([2.])
	elif numGP == 2:
		xi_GP = torch.tensor([-math.sqrt(1/3), math.sqrt(1/3)])
		weights = torch.tensor([1., 1.])
	elif numGP == 3:
		xi_GP = torch.tensor([-math.sqrt(0.6), 0, math.sqrt(0.6)])
		weights = torch.tensor([5./9, 8./9, 5./9])
	elif numGP == 4:
		xi_GP = torch.tensor([-0.861136, -0.339981, 0.339981, 0.861136])
		weights = torch.tensor([0.5-math.sqrt(30)/36, 0.5+math.sqrt(30)/36, 0.5+math.sqrt(30)/36, 0.5-math.sqrt(30)/36])
	elif numGP == 5:
		xi_GP = torch.tensor([-0.90618, -0.538469, 0, 0.538469, 0.90618])
		weights = torch.tensor([0.236927, 0.478629, 0.568889, 0.478629, 0.236927])
	else:
		print("Error. Only upto 5 Gaussian Points are allowed.")

	return [xi_GP, weights]

def getGlobalMapping(x, y, numGP_x, numGP_y):
	""" Maps the local coordinates to global coordinates using linear shape functions
		and returns the corresponding global coordinates, global weights and jacobian """
	# In each direction, number of intervals = number of nodes - 1
	numIntervals_x = x.size(dim=0)-1	
	numIntervals_y = y.size(dim=0)-1 
	numXi = numIntervals_x*numGP_x
	numEta = numIntervals_y*numGP_y
	global_xi = torch.zeros(numXi)
	global_eta = torch.zeros(numEta)
	global_weights_xi = torch.zeros(numXi)
	global_weights_eta = torch.zeros(numEta)
	jacobian_xi = torch.zeros(numXi)
	jacobian_eta = torch.zeros(numEta)

	[xi_GP, xi_w] = getGaussianVariables(numGP_x)
	[eta_GP, eta_w] = getGaussianVariables(numGP_y)
	
	for el in range(numIntervals_x):
		# mapping local (xi) to global (x) using linear shape functions
		A = (x[el+1] - x[el])*0.5
		B = (x[el+1] + x[el])*0.5
		for i in range(numGP_x):
			index = numGP_x*el+i
			global_xi[index] = A*xi_GP[i] + B
			global_weights_xi[index] = xi_w[i]
			jacobian_xi[index] = A
	for el in range(numIntervals_y):
		# mapping local (eta) to global (y) using linear shape functions
		A = (y[el+1] - y[el])*0.5
		B = (y[el+1] + y[el])*0.5
		for i in range(numGP_y):
			index = numGP_y*el+i
			global_eta[index] = A*eta_GP[i] + B
			global_weights_eta[index] = eta_w[i]
			jacobian_eta[index] = A
	global_xi, global_eta = torch.meshgrid(global_xi, global_eta)
	
	# reshaping is required since model takes tensor of shape(n,2)
	global_map = torch.cat((global_xi.reshape(-1,1), global_eta.reshape(-1,1)), 1)
	global_weights_xi, global_weights_eta = torch.meshgrid(global_weights_xi, global_weights_eta)
	global_weights = torch.cat((global_weights_xi.reshape(-1,1), global_weights_eta.reshape(-1,1)), 1)
	jacobian_xi, jacobian_eta = torch.meshgrid(jacobian_xi,jacobian_eta)
	jacobian = (jacobian_xi*jacobian_eta).reshape(-1,1)
	return global_map, global_weights, jacobian

def gaussianIntegration(integrand, weights, jacobian):
	# Performs gaussian integration based on the values of integrand, weights and jacobian
	integral = torch.sum(integrand*jacobian*(weights[:,0]*weights[:,1]).view(-1,1))
	return integral