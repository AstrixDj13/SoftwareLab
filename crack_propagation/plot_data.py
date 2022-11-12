import matplotlib.pyplot as plt
import numpy as np

def plotData(x, y, u, v, s, u_analytical, v_analytical, epochData, costData, trainingError, validationError):
	fig, ax = plt.subplots(2,2)
	for row in range(2):
	    ax[row,0].set_xlabel("x")
	    ax[row,0].set_ylabel("y")

	ax[0,0].set_title("u")
	cb = ax[0,0].pcolormesh(x.detach(), y.detach(), u.detach(), cmap='viridis', antialiased=False)
	ax[0,0].set_aspect('equal', adjustable='box')
	fig.colorbar(cb, ax = ax[0,0])

	ax[1,0].set_title("v")
	cb = ax[1,0].pcolormesh(x.detach(), y.detach(), v.detach(), cmap='viridis', antialiased=False)
	fig.colorbar(cb, ax = ax[1,0])
	ax[1,0].set_aspect('equal', adjustable='box')

	ax[0,1].set_title("Training History")
	ax[0,1].set_xlabel("Epochs")
	ax[0,1].set_ylabel("Cost Function")
	ax[0,1].semilogy(epochData, costData)

	# ax[1,1].set_title("Phase Field")
	# cb = ax[1,1].pcolormesh(x.detach(), y.detach(), s.detach(), cmap='plasma', antialiased=False)
	# fig.colorbar(cb, ax = ax[1,1])
	# ax[1,1].set_aspect('equal', adjustable='box')
	ax[1,1].set_title("Errors")
	ax[1,1].set_xlabel("Epochs")
	ax[1,1].set_ylabel("Error")
	ax[1,1].semilogy(epochData, validationError, color = 'r', label = "Validation Error")
	ax[1,1].semilogy(epochData, trainingError, color = 'b', label = "Training Error")
	ax[1,1].legend()

	fig.tight_layout()
	plt.show()

	error_u = abs(u.detach()-u_analytical.detach())
	error_v = abs(v.detach()-v_analytical.detach())

	fig2, ax = plt.subplots(2,2)
	for row in range(2):
		for col in range(2):
	 	   	ax[row,col].set_xlabel("x")
	 	   	ax[row,col].set_ylabel("y")

	ax[0,0].set_title("$u_{analytical}$")
	cb = ax[0,0].pcolormesh(x.detach(), y.detach(), u_analytical.detach(), cmap='viridis', antialiased=False)
	ax[0,0].set_aspect('equal', adjustable='box')
	fig2.colorbar(cb, ax = ax[0,0])

	ax[1,0].set_title("$v_{analytical}$")
	cb = ax[1,0].pcolormesh(x.detach(), y.detach(), v_analytical.detach(), cmap='viridis', antialiased=False)
	fig2.colorbar(cb, ax = ax[1,0])
	ax[1,0].set_aspect('equal', adjustable='box')

	ax[0,1].set_title("Error: u")
	cb = ax[0,1].pcolormesh(x.detach(), y.detach(), error_u, cmap='coolwarm', antialiased=False)
	fig2.colorbar(cb, ax = ax[0,1])
	ax[0,1].set_aspect('equal', adjustable='box')


	ax[1,1].set_title("Error: v")
	cb = ax[1,1].pcolormesh(x.detach(), y.detach(), error_v, cmap='coolwarm', antialiased=False)
	fig2.colorbar(cb, ax = ax[1,1])
	ax[1,1].set_aspect('equal', adjustable='box')

	fig2.tight_layout()
	plt.show()