# Implementation of the linear benchmark optimal policy from Proposition 1 

import numpy as np

def coefficients(eta, kappa, nu, sigma, N):
        
        a = np.zeros(N+1)
        b = np.zeros(N+1)
        c = np.zeros(N+1)
        d = np.zeros(N+1)
        rho = np.zeros(N+1)
        
        # index runs from 0 to N
        a[N] = eta/2
        b[N] = 1-kappa
        c[N] = 0
        d[N] = 0
        rho[N] = 2 * eta + 4 * nu + 4 * a[N] - 4 * eta * b[N] + 4 * eta**2 * c[N] 
        
        for i in range(N-1,0,-1):
            a[i] = a[i+1] + nu - 1/(rho[i+1]) * (2 * a[i+1] + 2 * nu - eta * b[i+1])**2
            b[i] = (1-kappa) * b[i+1] + 2 * (1-kappa)/(rho[i+1]) \
                    * (2 * a[i+1] + 2 * nu - b[i+1] * eta) * (1 - b[i+1] + 2 * eta * c[i+1])
            c[i] = c[i+1] * (1-kappa)**2 - (1-kappa)**2/(rho[i+1]) * (1 - b[i+1] + 2 * eta * c[i+1])**2 
            
            d[i] = c[i+1] * sigma**2 + d[i+1]
            
            rho[i] = 2 * eta + 4 * nu + 4 * a[i] - 4 * eta * b[i] + 4 * eta**2 * c[i]
            
        return a, b, c, d, rho
    
    
def optimizer(X0, d0, eta, kappa, nu, sigma, N, epsilon):
    
        control = np.zeros(N+1) 
        deviation = np.zeros(N+1)
        remainOrder = np.zeros(N+1)
        value = np.zeros(N+1)
        remainOrder[0] = X0
        deviation[0] = d0
        a, b, c, d, rho = coefficients(eta, kappa, nu, sigma, N) # indexed from 0 to N
        
        for i in range(1,N,1):
            
            control[i] = 2/(rho[i+1]) * ((2 * a[i+1] + 2 * nu - b[i+1] * eta) * remainOrder[i-1] - 
                                           (1 - b[i+1] + 2 * eta * c[i+1]) * (1-kappa) * deviation[i-1])
            
            remainOrder[i] = remainOrder[i-1] - control[i]
            deviation[i] = (1-kappa) * deviation[i-1] + eta * control[i] + epsilon[i]
            value[i] = a[i] * remainOrder[i-1]**2 + b[i] * remainOrder[i-1] * deviation[i-1] \
                        + c[i] * deviation[i-1]**2 + d[i] 
        
        control[N] = remainOrder[N-1]
        remainOrder[N] = remainOrder[N-1] - control[N]  
        deviation[N] = (1-kappa) * deviation[N-1] + eta * control[N] + epsilon[N]
        value[N] = a[N] * remainOrder[N-1]**2 + b[N] * remainOrder[N-1] * deviation[N-1] \
                    + c[N] * deviation[N-1]**2 + d[N]
        
        return control, remainOrder, deviation, value
    

def valueFunction(timeStep, remainOrder, deviation, eta, kappa, nu, sigma, N):
    
        a, b, c, d, rho = coefficients(eta, kappa, nu, sigma, N) # indexed from 0 to N
        
        return a[timeStep] * remainOrder**2 + b[timeStep] * remainOrder * deviation \
                + c[timeStep] * deviation**2 + d[timeStep]
    
    
def optimalPolicy(timeStep, remainOrder, deviation, eta, kappa, nu, sigma, N):
        
        a, b, c, d, rho = coefficients(eta, kappa, nu, sigma, N) # indexed from 0 to N
        
        if timeStep == N:
        
            return remainOrder
        
        else: 
        
            return 2/(rho[timeStep+1]) * ((2 * a[timeStep+1] + 2 * nu - b[timeStep+1] * eta) * remainOrder 
                                          - (1 - b[timeStep+1] + 2 * eta * c[timeStep+1]) * (1-kappa) * deviation) 
        
        
        
        
        
