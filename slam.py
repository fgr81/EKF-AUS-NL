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
import pandas as pd



class Slam:
    
    FILENAME_OUTPUT = 'out/output'
    FILENAME_TRAJECTORY = 'trajectory.dat'  
    # non va perche PrepareForAnalysis resituisce un xa con 7 colonne NC = 8  # Numero colonne di xa, DEVE coincidere con: long TotNumberPert(){ return m + 2 * ml * (ml +1)/2;}
    NC = 7
    SIGMA_ESTIMATE = 0.50
    SIGMA_STEERING = 0.01
    
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.phi = 0.
        self.v = 10.
        self.g = 0.
        self.lm = pd.DataFrame(data=None, columns=['idd', 'x','y','xa_x','xa_y'])
        
        self.album_misure = []
        # self.x_xa = np.random.normal(self.x, Slam.SIGMA_ESTIMATE, Slam.NC)
        self.x_xa = np.zeros(Slam.NC, dtype=float, order='F')
        # self.y_xa = np.random.normal(self.y, Slam.SIGMA_ESTIMATE, Slam.NC)
        self.y_xa = np.zeros(Slam.NC, dtype=float, order='F')
        # self.phi_xa = np.random.normal(self.phi, Slam.SIGMA_ESTIMATE, Slam.NC)
        self.phi_xa = np.zeros(Slam.NC, dtype=float, order='F')
        #self.v_xa = np.random.normal(self.v, Slam.SIGMA_ESTIMATE, Slam.NC)
        #self.g_xa = np.random.normal(self.g, Slam.SIGMA_ESTIMATE, Slam.NC)
        self.v_xa = np.random.normal(Slam.SIGMA_ESTIMATE, 0.01, Slam.NC)
        self.g_xa = np.random.normal(Slam.SIGMA_STEERING, 0.001, Slam.NC)
    
    def write_output_state_to_file(self, step = 0):
        f_stato = open(f"{Slam.FILENAME_OUTPUT}_{step}.dat", 'w')        
        f_stato.write(f"{self.x} ")
        for i in self.x_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.y} ")
        for i in self.y_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.phi} ")
        for i in self.phi_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.v} ")
        for i in self.v_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.g} ")
        for i in self.g_xa:
            f_stato.write(f"{i} ")  
        for index, row in self.lm.iterrows():
            f_stato.write(f"\n{row['idd']} {row['x']} {row['y']} ")                        
            for tt in row['xa_x']:                
                f_stato.write(f"{tt} ")
            for tt in row['xa_y']:    
                f_stato.write(f"{tt} ")                
        f_stato.close()
    
    
    def absoluting(self, meas):
        #rad = math.radians(self.phi)
        rad = self.phi
        
        # abs_x = self.x * math.cos(rad) - self.y * math.sin(rad)
        # abs_y = self.x * math.sin(rad) + self.y * math.cos(rad)
        # abs_x += meas['x']
        # abs_y += meas['y']
        
        _x = meas['x'] + self.x
        _y = meas['y'] + self.y
        abs_x =   _x * math.cos(rad) - _y * math.sin(rad)
        abs_y =   _x * math.sin(rad) + _y * math.cos(rad)
        
        return abs_x, abs_y
    
           
    def initial_assimilation(self, measure,t=0):
        for i in range(0,len(measure),3):        
            _id = int(measure[i])
            m = {'x': measure[i+1], 'y':measure[i+2]}
            abs_x, abs_y = self.absoluting(m)  
            # if _id == 0:
            #     print(f"kiki initial_assimilation meas_x:{m['x']} meas_y:{m['y']}   abs_x:{abs_x}  abs_y:{abs_y}")
            xa_x = np.random.normal(Slam.SIGMA_ESTIMATE, 0.01, Slam.NC)
            xa_y = np.random.normal(Slam.SIGMA_ESTIMATE, 0.01, Slam.NC)                
            #xa_x = np.random.normal(abs_x, Slam.SIGMA_ESTIMATE, Slam.NC)
            #xa_y = np.random.normal(abs_y, Slam.SIGMA_ESTIMATE, Slam.NC)                
            if (_id in set(self.lm['idd'])):
                mm = self.lm.loc[self.lm['idd'] == _id]
                mm = {'idd':_id, 'x':abs_x, 'y':abs_y, 'xa_x':xa_x, 'xa_y':xa_y}                
            else:
                mm = {'idd':_id, 'x':abs_x, 'y':abs_y, 'xa_x':xa_x, 'xa_y':xa_y}
                self.lm = self.lm.append(mm, ignore_index=True)

        
    def evolve(cols, timestep=0.1):
        DT = 0.025
        B = 2.71  # Viene dalla vettura di KITTI (dovrebbe essere la lunghezza)
        out = np.ones( cols.shape, dtype=float, order='F') 
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
                # x = x + ( v * DT * math.cos(math.radians(g + phi)) )
                # y = y + ( v * DT * math.sin(math.radians(g + phi)) )
                # phi = phi + ( v * DT/B * math.sin(math.radians(g)) )                
                x = x + ( v * DT * math.cos(g + phi) )
                y = y + ( v * DT * math.sin(g + phi) )
                phi = phi + ( v * DT/B * math.sin(g) )                
                #print(f"kiki evolve phi:{phi} con g:{g} e v:{v}")
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
            _id = self.album_misure[t][i]
            if _id in self.album_misure[t+1]:
                lm = {'x': state_col[3 + i*2], 'y':state_col[4 + i*2]}
                dist = TestBed.distanza_cartesiana(pto,angolo,lm)  # 'x','y'
                _out_mis.append(dist['x'])
                _out_mis.append(dist['y'])
        out_mis = np.zeros((len(_out_mis)), dtype=float, order='F')
        out_mis.flags.writeable = True
        for i in range(len(_out_mis)):
             out_mis[i] = _out_mis[i]
        return out_mis
    
    
    def give_xa_and_analysis(self, t):
       numero_di_landmark_misurati = len(self.album_misure[t]) 
       xa = np.ones( [numero_di_landmark_misurati*2 + 5, Slam.NC], dtype=float, order='F') 
       analysis = np.ones( [numero_di_landmark_misurati*2 + 5,1], dtype=float, order='F')
       xa[0,:] = self.x_xa
       xa[1,:] = self.y_xa
       xa[2,:] = self.phi_xa
       xa[3 + (numero_di_landmark_misurati*2),:] = self.v_xa
       xa[4 + (numero_di_landmark_misurati*2),:] = self.g_xa
       analysis[0] = self.x
       analysis[1] = self.y
       analysis[2] = self.phi
       analysis[3 + (numero_di_landmark_misurati*2)] = self.v
       analysis[4 + (numero_di_landmark_misurati*2)] = self.g
       for i in range(0,numero_di_landmark_misurati):
            _id = self.album_misure[t][i]
            xa[3 + i*2] = self.lm.loc[self.lm['idd'] == _id]['xa_x'].head(1).to_numpy()[0]
            xa[4 + i*2] = self.lm.loc[self.lm['idd'] == _id]['xa_y'].head(1).to_numpy()[0]
            analysis[3 + i*2] = self.lm.loc[self.lm['idd'] == _id]['x'].head(1)
            analysis[4 + i*2] = self.lm.loc[self.lm['idd'] == _id]['y'].head(1)
       return analysis, xa
   
    
    def update(self, analysis, xa, new_lms):
        self.x = analysis[0][0]
        self.y = analysis[1][0]       
        self.phi = analysis[2][0]       
        self.v = analysis[len(analysis) - 2][0]       
        self.g = analysis[len(analysis) - 1][0]
        self.x_xa = xa[0,:]
        self.y_xa = xa[1,:]
        self.phi_xa = xa[2,:]
        self.v_xa = xa[len(analysis) - 2,:]
        self.g_xa = xa[len(analysis) - 1,:]
        t = len(self.album_misure) - 2
        #
        for i in range(len(self.album_misure[t])):
           _id = self.album_misure[t][i]          
           self.lm.loc[self.lm['idd'] == _id].head(1)['x']= analysis[3 + 2*i]
           self.lm.loc[self.lm['idd'] == _id].head(1)['y'] = analysis[4 + 2*i]
           self.lm.loc[self.lm['idd'] == _id].head(1)['xa_x'][0] = xa[3 + 2*i,:]  
           self.lm.loc[self.lm['idd'] == _id].head(1)['xa_y'][0] = xa[4 + 2*i, :]
        #
        tt = int(len(new_lms)/3)
        for i in range(tt):
            _idd = new_lms[i*3]
            m = {'x': new_lms[i*3 + 1], 'y':new_lms[i*3 + 2]}
            abs_x, abs_y = self.absoluting(m)
            '''
            Se prendo un lm che è presente nello stato 
            (ma non all'istante precedente!  -> MISURE INTERMITTENTI
             come mi devo comportare?
             kiki todo parlarne con Luigi 
             '''
            # self.lm = self.lm.drop(self.lm[self.lm.idd == _idd].index)
            if len(self.lm.loc[self.lm['idd'] == _idd]) > 0:                
                xa_x = self.lm.loc[self.lm['idd'] == _idd].head(1)['xa_x'].to_numpy()[0]
                xa_y = self.lm.loc[self.lm['idd'] == _idd].head(1)['xa_y'].to_numpy()[0]             
                self.lm = self.lm.drop(self.lm[self.lm.idd == _idd].index)
            else:
                xa_x = np.random.normal(Slam.SIGMA_ESTIMATE, 0.01, Slam.NC)
                xa_y = np.random.normal(Slam.SIGMA_ESTIMATE, 0.01, Slam.NC)                
            new_lm = {'idd':_idd, 'x':abs_x, 'y':abs_y, 'xa_x':xa_x, 'xa_y':xa_y}
            self.lm = self.lm.append(new_lm, ignore_index=True)

            
        
def main():
    BEGIN_STEP = 0
    STOP_STEP = 45
    N = 5
    M = 6  # LinM
    P = 0
    ML = 1  # nonLinM
    slam = Slam()  # Inizializzo il sistema con i valori di default
    ekf = Ekf(N, M, P, ML)
    
    traiettoria = open(Slam.FILENAME_TRAJECTORY, 'w')
    
    measure, dummy, dummy2 = TestBed.get_measure(BEGIN_STEP,slam.album_misure)
    slam.initial_assimilation(measure)
    analysis, xa = slam.give_xa_and_analysis(0)
    slam.write_output_state_to_file(BEGIN_STEP)
        
    for i in range(BEGIN_STEP + 1, STOP_STEP):
        print(f"Nuovo step:{i}")        
        measure, vect_track_meas, new_lms = TestBed.get_measure(i, slam.album_misure)
        _analysis, _xa = ekf.worker(analysis,xa,vect_track_meas, Slam.evolve, slam.non_lin_h)
        slam.update(_analysis, _xa, new_lms)
        analysis, xa = slam.give_xa_and_analysis(i)
        slam.write_output_state_to_file(i)
        traiettoria.write(f"{slam.x} {slam.y} {slam.phi} {slam.v} {slam.g}\n")
        traiettoria.flush()
    traiettoria.close()


    
        

if __name__ == "__main__":
    main()
    print("finito, pace e bene.")   
    
