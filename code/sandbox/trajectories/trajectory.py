
import math
import numpy as np
import matplotlib.pyplot as plt
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann, mean_logeuclid

def project_sources(sources, mixture):
    s = sources
    A = mixture
    Nt,_,Ns = s.shape
    m,_ = A.shape
    x = np.zeros((Nt, m, Ns))
    for t in range(Nt):
        for n in range(Ns):
            x[t,:,n] = np.dot(A, s[t,:,n]) + 0.5*np.random.randn(m)
    return x    

def sigmoid(x, a, b):
    return 1.0/(math.exp(-a*(x-b)) + 1)

def gen_mvar(coeffs, Ns, Nt, sig=1.0, Nb=1000):
    
    if type(coeffs[0]) is list:
        m,m = coeffs[0][0].shape
        P = len(coeffs[0])     
        coeffs = [np.copy(coeffs[0]) for _ in range(Nb)] + coeffs 
    else:
        m,m = coeffs[0].shape
        P  = len(coeffs)
        coeffs = [np.copy(coeffs) for _ in range(Ns+Nb)] 
               
    x  = np.zeros((Nt,m,Ns+Nb))
    u  = np.random.randn(Nt,m,Ns+Nb)    
    for t in range(Nt):
        for n in range(Ns+Nb):
            for p in range(P):
                x[t,:,n] += np.dot(coeffs[n][p], x[t,:,n-(p+1)]) 
            x[t,:,n] += sig*u[t,:,n]
    x = x[:,:,Nb:] 
    return x   
    
def gen_windows(L, Ns, step=1):
    return np.stack([w + np.arange(L) for w in range(0,Ns-L+1, step)])    
        
r = 0.95
f = [0.1, 0.2, 0.3, 0.4]
Ns = 4096
Nt = 50
n  = len(f)

A1 = np.zeros((n,n))
for i,fi in enumerate(f):
    A1[i,i] = 2*r*np.cos(2*np.pi*fi)
A2 = -1*(r**2)*np.eye(n)

t   = np.linspace(-2,+2,Ns)
ev  = np.array(map(lambda x: sigmoid(x,a=50,b=-0.5), t))
ev -= np.array(map(lambda x: sigmoid(x,a=50,b=+0.5), t)) 
c10 =  0.90*ev
c12 =  0.75*ev
c32 = -0.80*ev

coeffs = [[np.copy(A1), np.copy(A2)] for _ in range(Ns)]
for i in range(Ns):
    coeffs[i][0][1,0] = c10[i]
    coeffs[i][0][1,2] = c12[i]
    coeffs[i][0][3,2] = c32[i]  

s = gen_mvar(coeffs, Ns, Nt, sig=1.0)

#%%

m = 16
Q =  np.random.randn(m,m)
Q = Q+Q.T
w,v = np.linalg.eig(Q)
A = v[:,:n]
x = project_sources(s, A)

#%%

L  = 128
st = 1
Cw = []
for w in gen_windows(L, Ns, step=st):
    xw = x[:,:,w]
    covs = Covariances().fit_transform(xw)
    Cw.append(np.mean(covs, axis=0))   

#%%          

from pyriemann.utils.distance import distance_riemann
stat  = Cw[:250]
Cstat = mean_riemann(np.stack(stat))
dist = []    
for Cwi in Cw:
    dist.append(distance_riemann(Cstat, Cwi))
plt.plot(L/2+np.arange(0, Ns-L+1),dist) 





    
    
    
    
    
    
    
    
    
    
    
    
    

