import numpy as np
from matplotlib import pyplot as plt
from PyQt5.QtCore import QThread
from scipy.signal import argrelextrema
from numba import njit
import time 

Pi = np.pi
    
class model(QThread):
    
    def __init__(self, epsilon, Z0, Z):
        super().__init__()
        self.S11 = S11(epsilon, Z0, Z)
        
    def calc_map(self, S11_min, Ca0, Cb0, Cab0, theta0):
        self.S11_min = S11_min
        self.Ca0 = Ca0
        self.Cb0 = Cb0
        self.Cab0 = Cab0
        self.theta0 = theta0
        self.start()
        
    def run(self):
        S11_min = self.S11_min
        Ca0 = self.Ca0
        Cb0 = self.Cb0
        Cab0 = self.Cab0
        theta0 = self.theta0
        Ca, Cb, Cab, theta = np.meshgrid(Ca0, Cb0, Cab0, theta0, indexing='ij') 
        S11 = self.S11(Ca, Cb, Cab, theta)
        self.argmin = S11.argmin(axis=3)
        theta_by_index = lambda i: theta0.min()+i*(theta0.max()-theta0.min())/(len(theta0)-1)
        self.theta_min = theta_by_index(self.argmin)
        self.min_w = np.sum(S11<=S11_min, axis=3)*(theta0.max()-theta0.min())/(len(theta0)-1)
        
    def afc(self, Ca, Cb, Cab, S11_min):
        v = self.S11.v
        R = Ca/Cb
        Z0ea = 1/(Ca*v)*1e12
        Z0eb = 1/(Cb*v)*1e12
        Z0oa = 1/((Ca+(1+R)*Cab)*v)*1e12
        Z0ob = 1/((Cb+((1+R)/R)*Cab)*v)*1e12
        theta = np.linspace(0.1, 180, num=5000)#degrees
        S11 = self.S11(Ca, Cb, Cab, theta)
        ind = argrelextrema(S11, np.less)[0]
        xs = theta[ind]
        ys = S11[ind]
        a = theta[np.where(S11<=S11_min)]
        if len(a)>0:
            dtheta = np.diff(theta).mean()
            ds = np.diff(a)
            
            ind_min = [[]]
            for i, d in enumerate(ds):
                if d<2*dtheta:
                    ind_min[-1].append(i)
                else:
                    ind_min[-1].append(i)
                    ind_min.append([])
            if len(ind_min[-1])>0:
                ind_min[-1].append(ind_min[-1][-1]+1)
        title =f'$C_a = {round(Ca)} pF, C_b = {round(Cb)} pF, '+'C_{ab}'
        title+=f' = {round(Cab)} pF, R = {round(R,1)}$\n'
        title+=f'$Z_e^a = {round(Z0ea,1)} \Omega, Z_e^b = {round(Z0eb,1)} \Omega, '
        title+=f'Z_o^a = {round(Z0oa,1)} \Omega, Z_o^b = {round(Z0ob,1)} \Omega $'
        
        annotations = []
        for i in range(len(ind)):
            x = xs[i]
            y = ys[i]
            if y<S11_min:
                annotation = f'$x = {round(x)}\degree$\n$y = {round(y,1)}\,dB$'
                if len(a)>0 and i<len(ind_min):
                    
                    t = a[ind_min[i]]
                    if len(t)>0:
                        x1 = t.min()
                        x2 = t.max()
                        dx = x2-x1
                    else:
                        dx = dtheta/2
                    annotation += f'\n$\Delta x = {round(dx,2)}\degree$'
                annotations.append(annotation)
            else:
                annotations.append('')
        return title, annotations, xs, theta, S11
        
class S11:
    c = 3e8 
    
    def __init__(self, epsilon, Z0=50, Z=5-15j):
        self.e = epsilon
        self.Z0 = Z0
        self.Z = Z
        self.v = self.c/np.sqrt(epsilon)
        self.times = []
        
    @staticmethod
    def read_from_table(fname='Mutual_capacitance.txt'):
        C = []
        with open(fname) as f:
            for line in f:
                if line[0]!='%':
                    while len(line.split(' '))>2:
                        line = line.replace('  ', ' ')
                    C.append(np.array(line.replace('\n', '').split(' '), 
                                      dtype=float))
        C = np.array(C)
        Ca = C[0,0]
        Cb = C[1,1]
        Cab = (C[1,0]+C[0,1])/2
        return Ca, Cb, Cab #pF
    
    def __call__(self, Ca, Cb, Cab, theta, dB=True):
        return self.evaluate(Ca, Cb, Cab, theta)
    
    def evaluate(self, Ca, Cb, Cab, theta, dB=True):#pF, pF, pF, degrees
        v = self.v
        Z = self.Z
        Z0 = self.Z0
        t = time.time()
        self.S11 = self._evaluate(Ca, Cb, Cab, theta, v, Z, Z0)
        self.times.append(time.time()-t)
        self.abs_S11 = self._abs(self.S11)
        if dB:
            self.abs_S11_dB = self._2dB(self.abs_S11)
            ans = self.abs_S11_dB
        else:
            self.abs_S11_dB = None
            ans = self.abs_S11
        return ans
    
    @staticmethod
    @njit(parallel=True)
    def _evaluate(Ca, Cb, Cab, theta, v, Z, Z0):
        Ca=Ca*1e-12
        Cb=Cb*1e-12
        Cab=Cab*1e-12
        R = Ca/Cb
        Z0ea = 1/(Ca*v)
        Z0eb = 1/(Cb*v)
        Z0oa = 1/((Ca+(1+R)*Cab)*v)
        #Z0ob = 1/((Cb+((1+R)/R)*Cab)*v)
        theta = theta/180 * Pi #to radians
        Y0ea = 1/Z0ea 
        Y0eb = 1/Z0eb
        Y0oa = 1/Z0oa 
        Delta = (Y0ea*1/np.sin(theta) - Y0oa*1/np.sin(theta))
        
        a1 = (Y0oa*1/np.tan(theta) + (Y0ea/R)*1/np.tan(theta))/Delta  
        b1 = 1j*(R + 1)/Delta
        c1 = 1j*(Y0oa**2 + Y0ea**2 - Y0oa*(Y0eb + R*Y0ea)*1/np.tan(theta)**2 - 
              2*Y0ea*Y0oa*1/np.sin(theta)**2)/((R + 1)*Delta)
        d1 = (R*Y0ea*1/np.tan(theta) + Y0oa*1/np.tan(theta))/Delta
        
        a2 = 1
        b2 = 0
        c2 = 1/Z
        d2 = 0
        
        a = a1*a2 + b1*c2
        b = a1*b2 + b1*d2
        c = c1*a2 + d1*c2
        d = c1*b2 + d1*d2
        
        S11 = (a + b/Z0 - c*Z0 - d)/(a + b/Z0 + c*Z0 + d)
        return S11
    
    @staticmethod
    @njit(parallel=True)
    def _abs(S11):
        return np.abs(S11)
    
    @staticmethod
    @njit(parallel=True)
    def _2dB(abs_S11):
        return 20*np.log10(abs_S11)