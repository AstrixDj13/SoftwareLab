import matplotlib.pyplot as plt

def plotData(x, y, u, v, s, epochData, costData, trainingError, validationError):
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
	ax[0,1].plot(epochData, costData)

	# ax[1,1].set_title("Phase Field")
	# cb = ax[1,1].pcolormesh(x.detach(), y.detach(), s.detach(), cmap='plasma', antialiased=False)
	# fig.colorbar(cb, ax = ax[1,1])
	# ax[1,1].set_aspect('equal', adjustable='box')
	ax[1,1].set_title("Errors")
	ax[1,1].set_xlabel("Epochs")
	ax[1,1].set_ylabel("Error")
	ax[1,1].plot(epochData, validationError, color = 'r', label = "Validation Error")
	ax[1,1].plot(epochData, trainingError, color = 'b', label = "Training Error")
	ax[1,1].legend()

	fig.tight_layout()
	plt.show()