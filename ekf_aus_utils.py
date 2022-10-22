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
    GMUNU_ESTIMATE = 1. 
    SIGMAD = 1.
    #SIGMAA = 0.25
    # QUESTA CLASSE DEVE POTER ESSERE USATA IN QUALSIASI SITUAZIONE PER CUI NON PUO CONTEMPLARE 
    # MECCANICHE INERENTI IL CASO D'USO 
    def __init__(self, n, m=6, p=0, ml=1, inflation=1.e-1 , l_factor=0.01):
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
        self.model_error = None
        self.r = None
        

    
              
        
    def worker(self, analysis, xa, meas, evolve, non_lin_h):
        """
       

        Parameters
        ----------
        analysis : TYPE
            DESCRIPTION.
        xa : TYPE
            DESCRIPTION.
        meas : array
            DESCRIPTION.
        evolve : TYPE
            DESCRIPTION.
        non_lin_h : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.gmunu = np.ones((len(analysis), 1), dtype=float, order='F') * EkfAusUtils.GMUNU_ESTIMATE
        self.gmunu.flags.writeable = True
        self.ekf.P(len(meas)) 
        # fmg 240222 dopo telef con LP  self.r = np.ones((self.ekf.P(), 1), dtype=float, order='F') * EkfAusUtils.SIGMAD * EkfAusUtils.SIGMAD
        self.r = np.ones((self.ekf.P(), 1), dtype=float, order='F') 
        for i in range(len(meas)):
            self.r[i] = EkfAusUtils.SIGMAD * EkfAusUtils.SIGMAD
            # if i == 0:
            #     self.r[i] = EkfAusUtils.SIGMAD * EkfAusUtils.SIGMAD
            # elif i % 2 == 1:
            #     self.r[i] = EkfAusUtils.SIGMAA * EkfAusUtils.SIGMAA
            # else:
            #     self.r[i] = EkfAusUtils.SIGMAD * EkfAusUtils.SIGMAD
        # self.r[-1, 0] = 1.e-4
        self.model_error = np.ones((2, 1), dtype=float, order='F') 
        self.model_error.flags.writeable = True
        self.model_error[0, 0] = 1.0 # 1.  # (m/s) error in the velocity
        self.model_error[1, 0] = 90 * math.pi/360.  #       90 * math.pi/360.  # 3 degrees error for the steering angle
        # self.ekf.SetModelErrorVariable(3, 4, self.model_error, xa)
        self.ekf.SetModelErrorVariable(len(analysis)-2, len(analysis)-1, self.model_error, xa)
        self.ekf.N(len(analysis))
        perturbazioni = self.ekf.PrepareForEvolution(analysis, xa, self.gmunu)
        perturbazioni = evolve(perturbazioni, 0.1)
        analysis = evolve(analysis, 0.1)
        xa = self.ekf.PrepareForAnalysis(analysis, perturbazioni, self.gmunu)  # kiki
        self.r = np.ones((len(meas), 1), dtype=float, order='F') * EkfAusUtils.SIGMAD * EkfAusUtils.SIGMAD
        self.r.flags.writeable = True
        measure = meas
        self.ekf.Assimilate(measure, non_lin_h, self.r, analysis, xa)
        return analysis, xa
    
