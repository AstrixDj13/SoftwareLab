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
    
    Normal_strain_energy = 0.5 * E(x,y)/(1 - nu(x,y)**2) * (eux**2 + evy**2 + 2*nu(x,y)*eux*evy)
    Shear_strain_energy = 0.25 * E(x,y)/(1 + nu(x,y))  * ((euy + evx)**2)
    IE_density = Normal_strain_energy + Shear_strain_energy
    int_energy = gaussian.gaussianIntegration(IE_density, weight, jacobian)
    
    return int_energy

def getElasticEnergy(u, x,y, s, weight, jacobian, E, nu, Gc, eps):
    eux = getDerivative(u[:,0].view(-1,1),x)
    evy = getDerivative(u[:,1].view(-1,1),y)
    euy = getDerivative(u[:,0].view(-1,1),y)
    evx = getDerivative(u[:,1].view(-1,1),x)
    # if s is continuous
    # sx =  getDerivative(s.view(-1,1),x)
    # sy =  getDerivative(s.view(-1,1),y)
    # if s is discontinuous
    # sx = 0
    # sy = 0 
    G = 0.5*E(x,y)/(1 + nu(x,y))
    K = G/(1 - 2*nu(x,y))
    eta = 0#1e-10
    phi_plus = 0.5*K*(((eux + evy - torch.abs(eux + evy))*0.5)**2)
    phi_minus = 0.5*K*(((eux + evy + torch.abs(eux + evy))*0.5)**2) + G*(eux**2 + evy**2 + 0.5*(euy + evx)**2)
    I1 = (s**2 + eta)*(phi_plus + phi_minus)
    # I2 = Gc*(0.25*((1 - s)**2)/eps + eps*(sx**2 + sy**2))
    Elatic_Energy_density = I1 
    elastic_energy = gaussian.gaussianIntegration(Elatic_Energy_density, weight, jacobian)
    
    return elastic_energy

def getExternalEnergy(u, x,y, weight, jacobian, p):
    EE1 = -p(x,y) * u[:,0].view(-1,1)
    EE2 = -p(x,y) * u[:,1].view(-1,1)
    EE_density = EE1 + EE2
    ext_energy = gaussian.gaussianIntegration(EE_density, weight, jacobian)
    
    return ext_energy


def costFunction(U, x,y, weight, jacobian, p, E, nu,s,Gc,eps):
    # return getExternalEnergy(U, x,y, weight, jacobian,p) + getInternalEnergy(U, x,y, weight, jacobian, E, nu) # w/o crack 
    return getExternalEnergy(U, x,y, weight, jacobian,p) + getElasticEnergy(U, x,y, s,weight, jacobian, E, nu, Gc, eps) # with crack
