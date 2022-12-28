"""
@author: debjit
"""
import torch
from torch.autograd import grad
import gaussian
import math
import matplotlib.pyplot as plt
import time

def getDerivative(y, x):
    dydx = grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    return dydx

def getNumericalDerivative(s, x, y):
    size = int(math.sqrt(y.size()[0]))
    sx = torch.zeros(size,size).to(torch.device("cuda"))
    sy = torch.zeros(size,size).to(torch.device("cuda"))
    # tic = time.time()
    for i in range(size):
        for j in range(size):
            if i == 0:
                sx[i,j] = (s[size*(i+1) + j] - s[size*i + j] )/(x[size*(i+1) + j] - x[size*i + j])
                if j == 0:
                    sy[i,j] = (s[(size*i + j) + 1] - s[size*i + j])/(y[j+1] - y[j])
                elif j == (size - 1):
                    sy[i,j] = (s[size*i + j] - s[(size*i + j) - 1])/(y[j] - y[j-1])
                else:
                    sy[i,j] = (s[(size*i + j) + 1] - 2*s[size*i + j] + s[(size*i + j) - 1])/(y[j+1] - y[j-1])  
            elif i == (size-1):
                sx[i,j] = (s[size*i + j] - s[size*(i-1) + j])/(x[size*i + j] - x[size*(i-1) + j])
                if j == (size-1):
                    sy[i,j] = (s[size*i + j] - s[(size*i + j) - 1])/(y[j] - y[j-1])
                elif j == 0:  
                    sy[i,j] = (s[(size*i + j) + 1] - s[size*i + j])/(y[j+1] - y[j])
                else:
                    sy[i,j] = (s[(size*i + j) + 1] - 2*s[size*i + j] + s[(size*i + j) - 1])/(y[j+1] - y[j-1])  
            elif j == 0:
                sx[i,j] = (s[size*(i+1) + j] - 2*s[size*i + j] + s[size*(i-1) + j])/(x[size*(i+1) + j] - x[size*(i-1) + j])
                sy[i,j] = (s[(size*i + j) + 1] - s[size*i + j])/(y[j+1] - y[j])
            elif j == (size-1):
                sx[i,j] = (s[size*(i+1) + j] - 2*s[size*i + j] + s[size*(i-1) + j])/(x[size*(i+1) + j] - x[size*(i-1) + j])
                sy[i,j] = (s[size*i + j] - s[(size*i + j) - 1])/(y[j] - y[j-1])
            else:           
                sx[i,j] = (s[size*(i+1) + j] - 2*s[size*i + j] + s[size*(i-1) + j])/(x[size*(i+1) + j] - x[size*(i-1) + j])
                sy[i,j] = (s[(size*i + j) + 1] - 2*s[size*i + j] + s[(size*i + j) - 1])/(y[j+1] - y[j-1])   
    # n = 120
    # x = x.reshape(n,n)
    # y = y.reshape(n,n)   
    # s = s.reshape(n,n)
    # cb = plt.pcolormesh(x.cpu().detach(), y.cpu().detach(), s.cpu().detach(), cmap='plasma', antialiased=False)
    # plt.title("Initial Phase Field")
    # plt.colorbar(cb)
    # plt.show()
    # cb = plt.pcolormesh(x.cpu().detach(), y.cpu().detach(), sx.cpu().detach(), cmap='plasma', antialiased=False)
    # plt.title("Initial Xder-Phase Field")
    # plt.colorbar(cb)
    # plt.show()
    # cb = plt.pcolormesh(x.cpu().detach(), y.cpu().detach(), sy.cpu().detach(), cmap='plasma', antialiased=False)
    # plt.title("Initial Yder-Phase Field")
    # plt.colorbar(cb)
    # plt.show()
    # toc = time.time()
    # t = toc - tic
    print(f"Derivation Time:{t}")
    return sx.reshape(-1,1), sy.reshape(-1,1)        

def getInternalEnergy(u, x,y, weight, jacobian, E, nu):
    eux = getDerivative(u[:,0].view(-1,1),x)
    evy = getDerivative(u[:,1].view(-1,1),y)
    euy = getDerivative(u[:,0].view(-1,1),y)
    evx = getDerivative(u[:,1].view(-1,1),x)
    
    Normal_strain_energy = 0.5 * E(x,y)/(1 - nu(x,y)**2) * (eux**2 + evy**2 + 2*nu(x,y)*eux*evy)
    Shear_strain_energy = 0.25 * E(x,y)/(1 + nu(x,y))  * ((euy + evx)**2)
    IE_density = Normal_strain_energy + Shear_strain_energy
    int_energy = gaussian.gaussianIntegration(IE_density, weight, jacobian)
    return int_energy

def getElasticEnergy(u, x,y, s, weight, jacobian, E, nu, Gc, eps, ds):
    eux = getDerivative(u[:,0].view(-1,1),x)
    evy = getDerivative(u[:,1].view(-1,1),y)
    euy = getDerivative(u[:,0].view(-1,1),y)
    evx = getDerivative(u[:,1].view(-1,1),x)
    # if s is continuous
    sx =  getDerivative(ds.view(-1,1),x)
    sy =  getDerivative(ds.view(-1,1),y)
    #tic = time.time()
    #sx, sy = getNumericalDerivative(s,x, y)
    #toc = time.time()
    #t = toc - tic
    #print(f"Derivative fn call Time:{t}")
    # device = torch.device("cuda")
    # sx = sx.to(device)
    # sy = sy.to(device)    
    # if s is discontinuous
    # sx = 0
    # sy = 0 
    G = 0.5*E(x,y)/(1 + nu(x,y))
    K = G/(1 - 2*nu(x,y))
    eta = 1e-10 # artificial crack stiffness
    
    Normal_strain_energy = 0.5 * E(x,y)/(1 - nu(x,y)**2) * (eux**2 + evy**2 + 2*nu(x,y)*eux*evy)
    Shear_strain_energy = 0.25 * E(x,y)/(1 + nu(x,y))  * ((euy + evx)**2)
    IE_density = Normal_strain_energy + Shear_strain_energy

    # phi_plus = 0.5*K*(((eux + evy + torch.abs(eux + evy))*0.5)**2) + G*(eux**2 + evy**2 + 0.5*(euy + evx)**2)
    phi_minus = 0.5*K*(((eux + evy - torch.abs(eux + evy))*0.5)**2)
    phi_plus = IE_density - phi_minus
    I1 = (s**2 + eta)*phi_plus + phi_minus
    I2 = Gc*(0.25*((1 - s)**2)/eps + eps*(sx**2 + sy**2))
    # Elatic_Energy_density = I1 + I2
    pureElasticEnergy = gaussian.gaussianIntegration(I1, weight, jacobian)
    crackEnergy = gaussian.gaussianIntegration(I2, weight, jacobian)
    # elastic_energy = gaussian.gaussianIntegration(Elatic_Energy_density, weight, jacobian) 

    elastic_energy = pureElasticEnergy + crackEnergy
    return elastic_energy, pureElasticEnergy, crackEnergy

def getExternalEnergy(u, x,y, weight, jacobian, p):
    EE1 = -p(x,y) * u[:,0].view(-1,1)
    EE2 = -p(x,y) * u[:,1].view(-1,1)
    EE_density = EE1 + EE2
    ext_energy = gaussian.gaussianIntegration(EE_density, weight, jacobian)
    return ext_energy


def costFunction(U, x,y, weight, jacobian, p, E, nu,s,Gc,eps,ds):
    # return getExternalEnergy(U, x,y, weight, jacobian,p) + getInternalEnergy(U, x,y, weight, jacobian, E, nu) # w/o crack 
    elasticEnergy, pureElasticEnergy, crackEnergy = getElasticEnergy(U, x,y, s,weight, jacobian, E, nu, Gc, eps, ds) 
    return getExternalEnergy(U, x,y, weight, jacobian,p) + elasticEnergy, pureElasticEnergy, crackEnergy # with crack