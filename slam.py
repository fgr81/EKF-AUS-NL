#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:20:20 2022

@author: fgr81
"""

from test_bed_maker import TestBed
from ekf_aus_utils import EkfAusUtils as Ekf
import numpy as np
import math 


class Slam:
    
    FILENAME_OUTPUT = 'out/output'
    FILENAME_TRAJECTORY = 'trajectory.dat'  
    # non va perche PrepareForAnalysis resituisce un xa con 7 colonne NC = 8  # Numero colonne di xa, DEVE coincidere con: long TotNumberPert(){ return m + 2 * ml * (ml +1)/2;}
    # NC = 7  # todo 
    SIGMA_ESTIMATE = 1.
    SIGMA_STEERING = 1.  # in radianti
    
    def __init__(self, nc):
        self.car = self.Car(nc)
        self.lm = []
        self.album_misure = []
        self.album_stato = []
        self.nc = nc

    class Car:
        def __init__(self, nc):
            self.x = 0.
            self.y = 0.
            self.phi = 0
            self.v = 10.
            self.g = 0.
            self.x_xa = np.random.normal(0, Slam.SIGMA_ESTIMATE, nc)
            self.y_xa = np.random.normal(0, Slam.SIGMA_ESTIMATE, nc)
            self.phi_xa = np.random.normal(0, Slam.SIGMA_ESTIMATE, nc)
            self.v_xa = np.random.normal(0, Slam.SIGMA_ESTIMATE, nc)
            self.g_xa = np.random.normal(0, Slam.SIGMA_STEERING, nc)

    class Lm:
        def __init__(self, idd, meas_x, meas_y, abs_x, abs_y, nc):
            self.idd = idd
            self.meas_x = meas_x
            self.meas_y = meas_y
            self.abs_x = abs_x
            self.abs_y = abs_y
            self.xa_x = np.random.normal(0, Slam.SIGMA_ESTIMATE, nc)
            self.xa_y = np.random.normal(0, Slam.SIGMA_ESTIMATE, nc)                

    def write_output_state_to_file(self, step = 0):
        f_stato = open(f"{Slam.FILENAME_OUTPUT}_{step}.dat", 'w')        
        f_stato.write(f"{self.car.x} ")
        for i in self.car.x_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.y} ")
        for i in self.car.y_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.phi} ")
        for i in self.car.phi_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.v} ")
        for i in self.car.v_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.g} ")
        for i in self.car.g_xa:
            f_stato.write(f"{i} ")
        for lm in self.lm:
            f_stato.write(f"\n{lm.idd} {lm.abs_x} {lm.abs_y}")
            for tt in lm.xa_x:                
                f_stato.write(f"{tt} ")
            for tt in lm.xa_y:    
                f_stato.write(f"{tt} ")                
        f_stato.close()
    
    
    def absoluting(self, meas):
        rad = self.car.phi
        _x =   self.car.x * math.cos(rad) + self.car.y * math.sin(rad)
        _y =   -self.car.x * math.sin(rad) + self.car.y * math.cos(rad)
        __x = meas['x'] + _x
        __y = meas['y'] + _y
        abs_x =  __x * math.cos(rad) - __y * math.sin(rad)
        abs_y =  __x * math.sin(rad) + __y * math.cos(rad)
        
        return abs_x, abs_y
        

    def initial_assimilation(self, measure,t=0):
        self.album_misure.append([])
        self.album_stato.append([])
        for i in range(0,len(measure),3):        
            _id = int(measure[i])
            m = {'x': measure[i+1], 'y':measure[i+2]}
            abs_x, abs_y = self.absoluting(m)  
            self.lm.append(self.Lm(_id,m['x'],m['y'],abs_x, abs_y, self.nc))
            self.album_misure[0].append(_id)
            self.album_stato[0].append(_id)

        
    def evolve(cols, timestep=0.1):
        DT = 0.025
        B = 2.71  # Viene dalla vettura di KITTI (dovrebbe essere la lunghezza)
        #out = np.ones( cols.shape, dtype=float, order='F') 
        # fmg 190722 nsteps = int(round(timestep / DT, 0))
        nsteps = 1
        DT = timestep
        #
        n_cols = cols.shape[1]
        for i in range(n_cols):
            x = cols[0,i]
            y = cols[1,i]
            phi = cols[2,i]
            v = cols[-2,i]
            g = cols[-1,i]                
            for n in range(nsteps):            
                x = x + ( v * DT * math.cos(g + phi) )
                y = y + ( v * DT * math.sin(g + phi) )
                phi = phi + ( v * DT/B * math.sin(g) )                

            cols[0,i] = x
            cols[1,i] = y
            cols[2,i] = phi  
        return cols
    
    
    def non_lin_h(self, state_col):
        '''
        Parameters
        ----------
        state_col : TYPE
            x,y,phi,v,g,[lm_x, lm_y]

        Returns
        -------
            Un vettore: ogni lm  nella colonna-stato, viene capito qual è il suo id, se è in misura, e nel caso qual è la misura

        '''
        _out_mis = []
        pto = {'x': state_col[0], 'y': state_col[1]}
        angolo = state_col[2]
        
        n = int((len(state_col) - 5 )/2)
        t = len(self.album_misure) - 2  # 
        for i in range(n):
            _id = self.album_stato[t][i]
            if _id in self.album_misure[t+1]:  # 
                lm = {'x': state_col[3 + i*2], 'y':state_col[4 + i*2]}
                dist = TestBed.distanza_cartesiana(pto,angolo,lm)  # 'x','y'
                _out_mis.append(dist['x'])
                _out_mis.append(dist['y'])
        out_mis = np.zeros((len(_out_mis)), dtype=float, order='F')
        out_mis.flags.writeable = True
        for idx,x in enumerate(_out_mis):
            out_mis[idx] = x
        return out_mis
    
    def give_measure(self, t):
        _out = []
        for _idd in self.album_misure[t]:
            # Cerco in self.lm il corrispondente
            for lm in self.lm:
                if lm.idd == _idd:
                    _out.append(lm.meas_x)
                    _out.append(lm.meas_y)
                    break
        meas = np.ones( (len(_out)), dtype=float, order='F')
        meas.flags.writeable = True
        for idx,x in enumerate(_out):
            meas[idx] = x
        return meas

    def give_xa_and_analysis(self, t):
        numero_di_landmark_misurati = len(self.album_stato[t]) 
        xa = np.ones( [numero_di_landmark_misurati*2 + 5, self.nc], dtype=float, order='F') 
        xa.flags.writeable = True
        analysis = np.ones( [numero_di_landmark_misurati*2 + 5,1], dtype=float, order='F')
        analysis.flags.writeable = True
        for idx, x in enumerate(self.car.x_xa):
            xa[0,idx] = x 
        for idx, x in enumerate(self.car.y_xa):
            xa[1,idx] = x
        for idx, x in enumerate(self.car.phi_xa):
            xa[2,idx] = x 
        for idx, x in enumerate(self.car.v_xa):
            xa[3 + (numero_di_landmark_misurati*2),idx] = x 
        for idx, x in enumerate(self.car.g_xa):
            xa[4 + (numero_di_landmark_misurati*2),idx] = x 
        analysis[0] = self.car.x
        analysis[1] = self.car.y
        analysis[2] = self.car.phi
        analysis[-2] = self.car.v
        analysis[-1] = self.car.g
        for i in range(0,numero_di_landmark_misurati):
            _id = self.album_stato[t][i]
            for lm in self.lm:
                if lm.idd == _id:
                    xa[3 + i*2] = lm.xa_x
                    xa[4 + i*2] = lm.xa_y
                    analysis[3 + i*2] = lm.abs_x
                    analysis[4 + i*2] = lm.abs_y
                    break
        return analysis, xa

    def update(self, analysis, xa):
        self.car.x = analysis[0][0]
        self.car.y = analysis[1][0]    
        self.car.phi = analysis[2][0]       
        self.car.v = analysis[len(analysis) - 2][0]
        self.car.g = analysis[len(analysis) - 1][0]
        self.car.x_xa = xa[0,:]
        self.car.y_xa = xa[1,:]
        self.car.phi_xa = xa[2,:]
        self.car.v_xa = xa[len(analysis) - 2,:]
        self.car.g_xa = xa[len(analysis) - 1,:]
        t = len(self.album_stato) - 2
        for i in range(len(self.album_stato[t])):
           _id = self.album_stato[t][i]          
           for  lm in self.lm:
               if lm.idd == _id:
                   lm.x = analysis[3 + 2*i][0]
                   lm.y = analysis[4 + 2*i][0]
                   lm.xa_x = xa[3 + 2*i, :]
                   lm.xa_y = xa[4 + 2*i, :]
                   break


def main():
    BEGIN_STEP = 0
    STOP_STEP = 100
    N = 5
    M = 6  # LinM
    P = 0
    ML = 4  # nonLinM
    ekf = Ekf(N, M, P, ML)
    slam = Slam(ekf.nc)  # Inizializzo il sistema con i valori di default
    
    traiettoria = open(Slam.FILENAME_TRAJECTORY, 'w')
    
    scan = TestBed.get_scan(BEGIN_STEP)
    slam.initial_assimilation(scan) 
    analysis, xa = slam.give_xa_and_analysis(0)
    slam.write_output_state_to_file(BEGIN_STEP)
        
    for i in range(BEGIN_STEP + 1, STOP_STEP):

        print(f"Nuovo step:{i}, slam.car(x,y,phi,v,g): {slam.car.x}, {slam.car.y}, {slam.car.phi}, {slam.car.v}, {slam.car.g}")
        
        slam.album_misure.append([])
        slam.album_stato.append([])
        
        scan = TestBed.get_scan(i)
        _n = int(len(scan)/3)

        for ii in range(_n):
            _id = scan[ii*3]
            slam.album_stato[i].append(_id)
            if _id in slam.album_stato[i-1]:  # è in tracking
                slam.album_misure[i].append(_id)

            trovato = 0
            for lm in slam.lm:
                if lm.idd == _id:
                    trovato = 1
                    lm.meas_x = scan[ii*3 + 1]
                    lm.meas_y = scan[ii*3 + 2]
                    break
            if trovato == 0:
                # Nuovo lm, viene aggiunto in self.lm
                m = {'x': scan[ii*3 + 1], 'y':scan[ii*3 + 2]}
                abs_x, abs_y = slam.absoluting(m)
                slam.lm.append(slam.Lm(idd=scan[ii*3], meas_x=scan[ii*3 +1], meas_y=scan[ii*3 +2], abs_x=abs_x, abs_y=abs_y, nc=slam.nc))
                
        measure = slam.give_measure(i)
        print("Misura len:", str(len(measure)/2))
        # todo kiki gestire measure==0
        _analysis, _xa = ekf.worker(analysis, xa, measure, Slam.evolve, slam.non_lin_h)        
        slam.update(_analysis, _xa)
        analysis, xa = slam.give_xa_and_analysis(i)

        slam.write_output_state_to_file(i)
        traiettoria.write(f"{slam.car.x} {slam.car.y} {slam.car.phi} {slam.car.v} {slam.car.g}\n")
        traiettoria.flush()

    traiettoria.close()


    
        

if __name__ == "__main__":
    main()
    print("finito, pace e bene.")   
    
