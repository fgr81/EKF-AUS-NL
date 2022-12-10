#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:31:23 2022

@author: fgr81


novità: distanza_cartesiana in get_measure
  nella misura iniziale il punto è preso da file, trasformato in misura-- si puo usare iniatial_assimilation per vedere se 
    il punto torna a essere nel valore preso da file (con la funzione slam.absoluting)
"""
    
    
import numpy as np    
import math
import secrets
import random
from numpy import ndarray


        
class TestBed:

    def __init__(self):
        print("Instanzio la classe TestBed")
    #####
    #
    #  
    #
    #####
    def raggio(primo, secondo):
        delta_x = secondo['x'] - primo['x']
        delta_y = secondo['y'] - primo['y']
        return ( delta_x ** 2 + delta_y ** 2) ** (1/2)
    
    def arc_tang_aggiustato(primo, secondo):
        delta_x = secondo['x'] - primo['x']
        delta_y = secondo['y'] - primo['y']
        '''if delta_x > 0:
            direz = math.atan(delta_y / delta_x)                
        elif delta_x < 0 and delta_y >= 0:
            direz = math.atan(delta_y / delta_x) + math.pi
        elif delta_x < 0 and delta_y < 0:
            direz = math.atan(delta_y / delta_x) - math.pi
        elif delta_x == 0 and delta_y > 0:
            direz = math.pi / 2
        elif delta_x == 0 and delta_y < 0:
            direz = -math.pi / 2
        else:
            direz = 0  # Non definita, non dovrebbe mai verificarsi che i due punti coincidano
        '''
        direz = math.atan2(delta_y, delta_x)
        direz_angolo = math.degrees(direz)
        return direz, direz_angolo
    
    def fai_percorso():
        ###
        # Ipotizzo che l'auto vada a 36 km/h = 10 m/sec 
        ###
        x = 0.
        y = 0.
        v = 10.
        #phi = 0.
        #t = 0.
        DT = 0.1  # 10 frame per second
        f_p = open("percorso.dat", "w")
        f_p.write(f"{x} {y}\n")
        segments = [{"l": 10, "steer": 0},
                    {"l": 10, "steer": 0},
                    {"l": 10, "steer": 20},
                    {"l": 10, "steer": 20},
                    {"l": 10, "steer": 20},
                    {"l": 10, "steer": 40},
                    {"l": 10, "steer": 40},
                    {"l": 10, "steer": 40},
                    {"l": 10, "steer": 20},
                    {"l": 10, "steer": 20},
                    ]
        for segment in segments:
            print("NUOVO SEGMENTO")
            l = segment['l']
            rad = math.radians(segment['steer'])
            while l > 0:
                dl = v * DT
                x += dl * math.cos(rad)
                y += dl * math.sin(rad)
                l -= dl
                _dv = secrets.choice(range(0, 10)) * 0.02
                if random.random() > 0.5:
                    v += _dv
                else:
                    v -= _dv
                if v < 1:
                    v = 1
                f_p.write(f"{x} {y}\n")
                print(f"v:{v} l:{l} x:{x} y:{y}")
        f_p.close()
            
    def posiziona_land_marks():
        file = open('percorso.dat', 'r')
        lines = file.readlines()
        file2 = open('landmarks.dat', 'w')
        _id = 0
        p = {'x': -1, 'y': -1}
        p_p = { 'x': 0, 'y': 0}  # punto precedente
        for line in lines:
            row = line.split()
            if p['x'] == -1:
               pass
            else:
                p_p = p 
            p['x'] = float(row[0])
            p['y'] = float(row[1])
            n_lm = secrets.choice(range(0,20))  # numero di landmark da posizionare 
            #n_lm = secrets.choice(range(0,10))  # numero di landmark da posizionare 
            for i in range(0,n_lm):
                # angolo = secrets.choice(range(0, 180))
                # fmg 210222 kk, angolo = TestBed.arc_tang_aggiustato(p, p_p)
                kk, angolo = TestBed.arc_tang_aggiustato(p_p, p)
                angolo += secrets.choice(range(-20,20))
                radianti = math.radians(angolo)
                lung = secrets.choice(range(0, 10))
                x_lm = p['x'] + lung*math.cos(radianti)
                y_lm = p['y'] + lung*math.sin(radianti)
                file2.write(f"{_id} {x_lm} {y_lm}\n")
                _id += 1
                print(f"lung:{lung} angolo:{angolo} x_lm:{x_lm} y_lm:{y_lm}")
        file2.close()
        file.close()
        
    def posiziona_land_marks_2():
        file = open('percorso.dat', 'r')
        file2 = open('landmarks.dat', 'w')
        lines = file.readlines()
        _x_min=100000
        _x_max=0
        _y_min = 10000
        _y_max = 0
        for line in lines:
            row = line.split()
            _x = float(row[0])
            _y = float(row[1])
            if _x < _x_min:
                _x_min = _x
            if _x > _x_max:
                _x_max = _x
            if _y < _y_min :
                _y_min = _y
            if _y > _y_max:
                _y_max = _y
        NUMERO_LM = 90000
        righe = int(math.sqrt(1000))
        delta_x = (_x_max-_x_min) / (righe + 1)
        delta_y = (_y_max-_y_min) / (righe +1)
        for i in range(NUMERO_LM):
            file2.write(f"{i} {_x_min+delta_x*(i % righe)} {_y_min+delta_y* (i // righe)}\n")
        file.close()
        file2.close()
            
            
      
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
    
    @staticmethod   
    def get_scan(t):
        global DEBUG 
        log = DEBUG['log_lm_idd']
        if log == 1:
            ff = open(f'utils/scans/scan_{t}','w')
            ff.write('-1 0 0\n')  # mi torna comodo per gnuplot, 
                                # voglio avere i grafici che visualizzini 
                                # l'origine (0,0), in modo da dare un senso di
                                # movimento
            
        out = []
        p = {'x':0., 'y':0.}
        p_p = {'x':0., 'y':0.}
        f_p = open('percorso.dat', 'r')
        lines = f_p.readlines()        
        line = lines[t]
        items = line.split()
        p['x'] = float(items[0])
        p['y'] = float(items[1])
        if t > 0:
            line_p = lines[t-1]            
            items_p = line_p.split()            
            p_p['x'] = float(items_p[0])
            p_p['y'] = float(items_p[1])
            radians, angolo = TestBed.arc_tang_aggiustato(p_p, p)            
        else:
            radians = angolo = 0.
        f_p.close()
        # fmg 191122 MAX_RAGGIO = 50.
        MAX_RAGGIO = 10.
        # fmg 191122 MAX_ANG = 20.
        MAX_ANG = 80.
        # Leggo dal file tutti i land mark presenti 
        file_lm = open('landmarks.dat', 'r')
        lines = file_lm.readlines()
        for line in lines:
            row = line.split()
            lm = {'idd':int(row[0]), 'x': float(row[1]), 'y': float(row[2])}
            _raggio = TestBed.raggio(p, lm)
            if _raggio < MAX_RAGGIO:
                dum, ang_con_lm = TestBed.arc_tang_aggiustato(p, lm) # kiki
                if abs(angolo - ang_con_lm) < MAX_ANG:
                    # lo prendo ! 
                    distanza = TestBed.distanza_cartesiana(p,radians,lm)  
                    out.append(lm['idd'])
                    out.append(distanza['x'])
                    out.append(distanza['y'])
                    if log == 1:
                        ff.write(f"{lm['idd']} {lm['x']} {lm['y']}\n")
        file_lm.close()
        if log == 1:
            ff.close()            
        return out
    
        
def main():
    TestBed.fai_percorso()
    TestBed.posiziona_land_marks_2()
    for i in range(100):
        print(i)
        TestBed.get_scan(i)
    
    print("finito, pace e bene.")        

DEBUG = {
    'log_lm_idd':1}

if __name__ == "__main__":
    main()
        
