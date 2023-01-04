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
    
    def __init__(self, ekf, car, SIGMA_ESTIMATE = 1., SIGMA_STEERING = 0.1):
        
        self.ekf = ekf
        nc = ekf.nc
        
        self.SIGMA_ESTIMATE = SIGMA_ESTIMATE
        self.SIGMA_STEERING = SIGMA_STEERING
        
        self.car = car
        
        self.x_xa = np.random.normal(0, SIGMA_ESTIMATE, nc)
        self.y_xa = np.random.normal(0, SIGMA_ESTIMATE, nc)
        self.phi_xa = np.random.normal(0, SIGMA_ESTIMATE, nc)
        self.v_xa = np.random.normal(0, SIGMA_ESTIMATE, nc)
        self.g_xa = np.random.normal(0, SIGMA_STEERING, nc)

        self.lm = []
        self.album_misure = []
        self.album_stato = []
        

            
    # class Lm:
    #     def __init__(self, idd, meas_x, meas_y, abs_x, abs_y, nc, SIGMA_ESTIMATE = 1.):
    #         self.idd = idd
    #         self.meas_x = meas_x
    #         self.meas_y = meas_y
    #         self.abs_x = abs_x
    #         self.abs_y = abs_y
    #         self.xa_x = np.random.normal(0, SIGMA_ESTIMATE, nc)
    #         self.xa_y = np.random.normal(0, SIGMA_ESTIMATE, nc)                

    def write_output_state_to_file(self, step = 0):
        f_stato = open(f"{Slam.FILENAME_OUTPUT}_{step}.dat", 'w')        
        f_stato.write(f"{self.car.x} ")
        for i in self.x_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.y} ")
        for i in self.y_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.phi} ")
        for i in self.phi_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.v} ")
        for i in self.v_xa:
            f_stato.write(f"{i} ")
        f_stato.write(f"\n{self.car.g} ")
        for i in self.g_xa:
            f_stato.write(f"{i} ")
        for lm in self.lm:
            f_stato.write(f"\n{lm.idd} {lm.abs_x} {lm.abs_y} ")
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
            self.lm.append(Lm(_id,m['x'],m['y'],abs_x, abs_y, self))
            self.album_misure[0].append(_id)
            self.album_stato[0].append(_id)

        
    def evolve(cols, timestep=0.1):
        DT = 0.025
        B = 2.71  # Viene dalla vettura di KITTI (dovrebbe essere la lunghezza)
        # fmg 190722 
        nsteps = int(round(timestep / DT, 0))
        #nsteps = 1
        #DT = timestep
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
    
    @staticmethod
    def distanza_cartesiana(p, rad,lm):
        '''
        Parameters
        ----------
        p : TYPE
            Punto dell'osservatore
        angolo : TYPE
            Direzione dell'osservatore
        lm : TYPE
            Landmark {x,y}
        Returns
        -------
        Distanza cartesiana {'x','y'} fra l'osservatore e il lm.
        
        '''
        # 210722 skype con luigi
        d_x = lm['x'] - p['x']
        d_y = lm['y'] - p['y']        
        x = d_x * math.cos(rad) + d_y * math.sin(rad)
        y = - d_x * math.sin(rad) + d_y * math.cos(rad)        
        
        dist = {'x':x, 'y': y}
        
        return dist
        
    
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
                dist = self.distanza_cartesiana(pto,angolo,lm)  # 'x','y'
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
        xa = np.ones( [numero_di_landmark_misurati*2 + 5, self.ekf.nc], dtype=float, order='F') 
        xa.flags.writeable = True
        analysis = np.ones( [numero_di_landmark_misurati*2 + 5,1], dtype=float, order='F')
        analysis.flags.writeable = True
        for idx, x in enumerate(self.x_xa):
            xa[0,idx] = x 
        for idx, x in enumerate(self.y_xa):
            xa[1,idx] = x
        for idx, x in enumerate(self.phi_xa):
            xa[2,idx] = x 
        for idx, x in enumerate(self.v_xa):
            xa[3 + (numero_di_landmark_misurati*2),idx] = x 
        for idx, x in enumerate(self.g_xa):
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
        self.x_xa = xa[0,:]
        self.y_xa = xa[1,:]
        self.phi_xa = xa[2,:]
        self.v_xa = xa[len(analysis) - 2,:]
        self.g_xa = xa[len(analysis) - 1,:]
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
    
    def alleggerisci_lms(self):        
        if len(self.album_stato) < 3 :
            return -1        
        last_idx = len(self.album_stato) -1 
        _cache = []
        for i in self.album_stato[last_idx]:
            _cache.append(i)
        for i in self.album_stato[last_idx - 1]:
            _cache.append(i)
        for lm in self.lm:
            if lm.idd in _cache:
                pass
            else:
                self.lm.remove(lm)
        
                
    def iterazione(self, i, scan):
        print(f"Nuovo step:{i}, self.car(x,y,phi,v,g): {self.car.x}, {self.car.y}, {self.car.phi}, {self.car.v}, {self.car.g}")
        analysis, xa = self.give_xa_and_analysis( i -1 )
        self.album_misure.append([])
        self.album_stato.append([])
        
        media_spost = {'x':0., 'y':0., 'cont':0}
        
        '''
        landmark scansionati per la prima volta, in questa iterazione  non
        partecipano all'assimilazione ma vengono conservati in memoria per la 
        prox iterzione
        '''
        nuovi_lm = []  
                       
        _n = int(len(scan)/3)
        for ii in range(_n):
            _id = scan[ii*3]
            self.album_stato[i].append(_id)
            if _id in self.album_stato[i-1]:  # è in tracking
                self.album_misure[i].append(_id)       
            trovato = 0
            for lm in self.lm:
                
                if lm.idd == _id:                   

                    trovato = 1

                    media_spost['cont'] += 1
                    media_spost['x'] += abs(lm.meas_x - scan[ ii*3 + 1])
                    media_spost['y'] += abs(lm.meas_y - scan[ ii*3 + 2])

                    lm.meas_x = scan[ii*3 + 1]
                    lm.meas_y = scan[ii*3 + 2]

                    break
                
            if trovato == 0:
                # Nuovo lm
                nuovi_lm.append(scan[ii*3])
                nuovi_lm.append(scan[ii*3 + 1])
                nuovi_lm.append(scan[ii*3 + 2])
                
        measure = self.give_measure(i)
        print("Misura len:", str(len(measure)/2))
        
        '''
        Nel caso in cui la misura sia nulla, bisogna passare allo step 
        successivo senza fare l'assimilazione
        '''
        if len(measure) > 0:
            print("Valor medio differenza misura, componente x:", media_spost['x']/media_spost['cont'], " componente y:", media_spost['y']/media_spost['cont'])
            
            _analysis, _xa = self.ekf.worker(analysis, xa, measure, Slam.evolve, self.non_lin_h)        
            self.update(_analysis, _xa)
        
        
        '''
        Ora che è stata assimilata la nuova posizione dell'auto, assimilo i 
        nuovi lm
        '''
        n_lm = int(len(nuovi_lm)/3)
        print('nuovi lm:', n_lm)
        for ii in range(n_lm):
            m = {'x': nuovi_lm[ii*3 + 1], 'y':nuovi_lm[ii*3 + 2]}
            abs_x, abs_y = self.absoluting(m)
            self.lm.append(Lm(idd=nuovi_lm[ii*3], meas_x=nuovi_lm[ii*3 +1], meas_y=nuovi_lm[ii*3 +2], abs_x=abs_x, abs_y=abs_y, slam=self))                            

        '''
        Elimino dalla memoria i lm vecchi in modo che il tempo di esecuzione sia
        pressochè costante
        '''        
        self.alleggerisci_lms()


class Car:
    def __init__(self, x = 0., y = 0., phi = 0., v = 10., g = 0.):
        self.x = x
        self.y = y
        self.phi = phi
        self.v = v
        self.g = g

    
class Lm():
    def __init__(self, idd, meas_x, meas_y, abs_x, abs_y, slam):
        self.idd = idd
        self.meas_x = meas_x
        self.meas_y = meas_y
        self.abs_x = abs_x
        self.abs_y = abs_y
        self.xa_x = np.random.normal(0, slam.SIGMA_ESTIMATE, slam.ekf.nc)
        self.xa_y = np.random.normal(0, slam.SIGMA_ESTIMATE, slam.ekf.nc)                
        
                

def main2():
    
    fornitore = TestBed()
    
    BEGIN_STEP = 0
    STOP_STEP = 124
    N = 5
    M = 6  # LinM
    P = 0
    ML = 4  # nonLinM
    model_error = np.ones((2, 1), dtype=float, order='F')   # todo il numero '2' deve arrivare da fuori parametricamente
    #model_error.flags.writeable = True
    model_error[0, 0] = 1. # 1.  # (m/s) error in the velocity
    model_error[1, 0] = 10 * math.pi/180.  #       90 * math.pi/360.  # 3 degrees error for the steering angle
    ekf = Ekf(N, M, P, ML, model_error)
    car = Car()
    slam = Slam(ekf, car)  # Inizializzo il sistema con i valori di default
    traiettoria = open(Slam.FILENAME_TRAJECTORY, 'w')
    
    scan = fornitore.get_scan(BEGIN_STEP)
    slam.initial_assimilation(scan) 
    analysis, xa = slam.give_xa_and_analysis(0)
    slam.write_output_state_to_file(BEGIN_STEP)
        
    for i in range(BEGIN_STEP + 1, STOP_STEP):
        scan = fornitore.get_scan(i)
        slam.iterazione(i, scan)
        slam.write_output_state_to_file(i)
        traiettoria.write(f"{slam.car.x} {slam.car.y} {slam.car.phi} {slam.car.v} {slam.car.g}\n")
        traiettoria.flush()

    traiettoria.close()

def main():
    BEGIN_STEP = 0
    STOP_STEP = 100
    N = 5
    M = 6  # LinM
    P = 0
    ML = 4  # nonLinM
    model_error = np.ones((2, 1), dtype=float, order='F')   # todo il numero '2' deve arrivare da fuori parametricamente
    #model_error.flags.writeable = True
    model_error[0, 0] = 1. # 1.  # (m/s) error in the velocity
    model_error[1, 0] = 10 * math.pi/180.  #       90 * math.pi/360.  # 3 degrees error for the steering angle
    ekf = Ekf(N, M, P, ML, model_error)
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
        
        ###
        #
        # 081222 festa dell'Immacolata, calcolo:
        # ' il valor medio sui lm della differenza fra la posizione (x,y) 
        # al tempo t e quella al tempo t-1'
        #
        ####
        media_spost = {'x':0., 'y':0., 'cont':0}
        
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
                    
                    # 081222
                    media_spost['cont'] += 1
                    media_spost['x'] += abs(lm.meas_x - scan[ ii*3 + 1])
                    media_spost['y'] += abs(lm.meas_y - scan[ ii*3 + 2])
                    #
                    
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
        #081222
        print("Valor medio differenza misura, componente x:", media_spost['x']/media_spost['cont'], " componente y:", media_spost['y']/media_spost['cont'])
        # todo kiki gestire measure==0
        _analysis, _xa = ekf.worker(analysis, xa, measure, Slam.evolve, slam.non_lin_h)        
        slam.update(_analysis, _xa)
        analysis, xa = slam.give_xa_and_analysis(i)

        slam.write_output_state_to_file(i)
        traiettoria.write(f"{slam.car.x} {slam.car.y} {slam.car.phi} {slam.car.v} {slam.car.g}\n")
        traiettoria.flush()

    traiettoria.close()


    
        

if __name__ == "__main__":
    main2()
    print("finito, pace e bene.")   
    
