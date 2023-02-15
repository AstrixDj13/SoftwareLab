"""
Cost or energy functions
"""
import torch
from torch.autograd import grad
import gaussian

def getDerivative(y, x):
    dydx = grad(y, x, torch.ones_like(x), create_graph=True, retain_graph=True)[0]
    return dydx

def getInternalEnergy(u, x, y, weight, jacobian, E, nu):
    # strain derivatives
    eux = getDerivative(u[:,0].view(-1,1),x)
    evy = getDerivative(u[:,1].view(-1,1),y)
    euy = getDerivative(u[:,0].view(-1,1),y)
    evx = getDerivative(u[:,1].view(-1,1),x)
    
    # energy calculations
    Normal_strain_energy = 0.5 * E(x,y)/(1 - nu(x,y)**2) * (eux**2 + evy**2 + 2*nu(x,y)*eux*evy)
    Shear_strain_energy = 0.25 * E(x,y)/(1 + nu(x,y))  * ((euy + evx)**2)
    IE_density = Normal_strain_energy + Shear_strain_energy
    int_energy = gaussian.gaussianIntegration(IE_density, weight, jacobian)
    return int_energy

def getEnergyWithCrack(u, x, y, s, weight, jacobian, E, nu, Gc, eps, ds):
    # strain derivatives
    eux = getDerivative(u[:,0].view(-1,1),x)
    evy = getDerivative(u[:,1].view(-1,1),y)
    euy = getDerivative(u[:,0].view(-1,1),y)
    evx = getDerivative(u[:,1].view(-1,1),x)
    
    # derivatives of crack field 
    sx =  getDerivative(ds.view(-1,1),x)
    sy =  getDerivative(ds.view(-1,1),y)
   
    # Material properties
    G = 0.5*E(x,y)/(1 + nu(x,y)) # shear modulus
    K = G/(1 - 2*nu(x,y)) # Bulk's modulus
    eta = 1e-10 # artificial crack stiffness
    
    Normal_strain_energy_density = 0.5 * E(x,y)/(1 - nu(x,y)**2) * (eux**2 + evy**2 + 2*nu(x,y)*eux*evy)
    Shear_strain_energy_density = 0.25 * E(x,y)/(1 + nu(x,y))  * ((euy + evx)**2)
    IE_density = Normal_strain_energy_density + Shear_strain_energy_density

    phi_minus = 0.5*K*(((eux + evy - torch.abs(eux + evy))*0.5)**2)
    phi_plus = IE_density - phi_minus
    I1 = (s**2 + eta)*phi_plus + phi_minus
    I2 = Gc*(0.25*((1 - s)**2)/eps + eps*(sx**2 + sy**2))

    # numerical integrations to compute energies from energy densities
    elasticEnergy = gaussian.gaussianIntegration(I1, weight, jacobian)
    crackEnergy = gaussian.gaussianIntegration(I2, weight, jacobian)
    totalEnergyWithCrack = elasticEnergy + crackEnergy

    return totalEnergyWithCrack, elasticEnergy, crackEnergy

def getExternalEnergy(u, x, y, weight, jacobian, p):
    EE1 = -p(x,y) * u[:,0].view(-1,1)
    EE2 = -p(x,y) * u[:,1].view(-1,1)
    EE_density = EE1 + EE2
    ext_energy = gaussian.gaussianIntegration(EE_density, weight, jacobian)
    return ext_energy

def costFunction(U, x,y, weight, jacobian, p, E, nu,s,Gc,eps,ds):
    ##--- w/o crack ---##
    # return getExternalEnergy(U, x, y, weight, jacobian,p) + getInternalEnergy(U, x, y, weight, jacobian, E, nu) 
    ##--- with crack ---##
    energyWithCrack, elasticEnergy, crackEnergy = getEnergyWithCrack(U, x, y, s,weight, jacobian, E, nu, Gc, eps, ds) 
    totalEnergy = energyWithCrack + getExternalEnergy(U, x, y, weight, jacobian,p)
    return  totalEnergy, elasticEnergy, crackEnergy 