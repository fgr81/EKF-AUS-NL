#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:48:05 2022

@author: fgr81
"""

from ekfaus import EkfAus
import numpy as np
import math

'''

l'ideale sarebbe che questa classe ricevesse come input unicamente i vettori di stato, di misuta, l'operatore H

'''


        
class EkfAusUtils:
    
    # QUESTA CLASSE DEVE POTER ESSERE USATA IN QUALSIASI SITUAZIONE PER CUI NON PUO CONTEMPLARE 
    # MECCANICHE INERENTI IL CASO D'USO 
    def __init__(self, n, m=6, p=0, ml=1, 
                 model_error = [],
                 inflation=1.e-1 , 
                 l_factor=0.01, 
                 SIGMA_R = 1., 
                 GMUNU_ESTIMATE = 1.):
        '''
        @n: lunghezza del vettore di stato
        @m:
        @p:
        @ml:
        '''
        self.ekf = EkfAus(n, m, p, ml)
        self.ekf.AddInflation(inflation)
        self.ekf.LFactor(l_factor)
        self.gmunu = None
        self.R = None
        # self.model_error = np.ones((2, 1), dtype=float, order='F')   # todo il numero '2' deve arrivare da fuori parametricamente
        # self.model_error.flags.writeable = True
        # self.model_error[0, 0] = 0.2 # 1.  # (m/s) error in the velocity
        # self.model_error[1, 0] = 5 * math.pi/180.  #       90 * math.pi/360.  # 3 degrees error for the steering angle
        self.model_error = model_error
        self.SIGMA_R = SIGMA_R
        self.GMUNU_ESTIMATE = GMUNU_ESTIMATE
        self.nc = self.ekf.linM() + self.ekf.HalfNumberNonLinPert()

                
        
    # def stampa_coda_Xa(self,Xa,k):
    #     print(k)
    #     # stampare ultime due righe 
    #     print(Xa[-2,:])
    #     print(Xa[-1,:])
       
    
    def worker(self, analysis, Xa, measure, evolve, non_lin_h):
        self.gmunu = np.ones(len(analysis), dtype=float, order='F') * self.GMUNU_ESTIMATE
        self.gmunu.flags.writeable = True
        self.ekf.P(len(measure)) 
        self.R = np.ones(len(measure), dtype=float, order='F') * (self.SIGMA_R ** 2)
        #self.R[-1] = 0.1
        self.R.flags.writeable = True
        #self.stampa_coda_Xa(Xa,'prima')
        self.ekf.SetModelErrorVariable(len(analysis)-2, len(analysis)-1, self.model_error, Xa)
        #self.stampa_coda_Xa(Xa,'dopo')
        self.ekf.N(len(analysis))

        perturbazioni = self.ekf.PrepareForEvolution(analysis, Xa, self.gmunu)
        perturbazioni = evolve(perturbazioni)
        analysis = evolve(analysis)
        Xa = self.ekf.PrepareForAnalysis(analysis, perturbazioni, self.gmunu)  
        meas = measure
        self.ekf.Assimilate(meas, non_lin_h, self.R, analysis, Xa)
        return analysis, Xa
    
