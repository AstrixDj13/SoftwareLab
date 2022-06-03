import torch
from torch.autograd import grad
import gaussian

def getDerivative(y, x):
    dydx = grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    return dydx

def getInternalEnergy(u, x,y, weight, jacobian, E, nu):
    eux = getDerivative(u[:,0].view(-1,1),x)
    evy = getDerivative(u[:,1].view(-1,1),y)
    euy = getDerivative(u[:,0].view(-1,1),y)
    evx = getDerivative(u[:,1].view(-1,1),x)
    
    Normal_strain_energy = (0.5 * E(x,y)/(1 - (nu(x,y)) ** 2)) * (eux * eux + evy * evy + ((2 * nu(x,y)) * eux * evy))
    Shear_strain_energy = (0.25 * E(x,y)/(1 + (nu(x,y))))  * ((euy + evx) ** 2)
    IE_density = Normal_strain_energy + Shear_strain_energy
    int_energy = gaussian.gaussianIntegration(IE_density, weight, jacobian)
    
    return int_energy


def getExternalEnergy(u, x,y, weight, jacobian, p):
    EE1 = -p(x,y) * u[:,0].view(-1,1)
    EE2 = -p(x,y) * u[:,1].view(-1,1)
    EE_density = EE1 + EE2
    ext_energy = gaussian.gaussianIntegration(EE_density, weight, jacobian)
    
    return ext_energy

def costFunction(U, x,y, weight, jacobian, p, E, nu):
    return getExternalEnergy(U, x,y, weight, jacobian,p) + getInternalEnergy(U, x,y, weight, jacobian, E, nu)
