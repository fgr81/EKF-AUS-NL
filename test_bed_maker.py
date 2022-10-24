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
        if delta_x > 0:
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
        direz_angolo = math.degrees(direz)
        return direz, direz_angolo
    
    def fai_percorso():
        ###
        # Ipotizzo che l'auto vada a 36 km/h = 10 m/sec 
        ###
        x = 0.
        y = 0.
        v = 10.
        phi = 0.
        t = 0.
        DT = 0.1  # 10 frame per second
        f_p = open("percorso.dat", "w")
        f_p.write(f"{x} {y}\n")
        segments = [{"l": 10, "steer": 0},
                    {"l": 10, "steer": 10},
                    {"l": 10, "steer": 30},
                    {"l": 10, "steer": 40},
                    {"l": 10, "steer": 50},
                    {"l": 100, "steer": 90},
                    {"l": 100, "steer": 120},
                    {"l": 100, "steer": 150},
                    {"l": 100, "steer": 180},
                    {"l": 100, "steer": 220},
                    {"l": 200, "steer": 240},
                    {"l": 600, "steer": 270},
                    {"l": 400, "steer": -45},
                    {"l": 200, "steer": -20},
                    {"l": 300, "steer": -10},
                    {"l": 200, "steer": 0},
                    {"l": 400, "steer": 60},
                    {"l": 1000, "steer": 90},
                    {"l": 1000, "steer": 120},
                    {"l": 1000, "steer": 90},
                    {"l": 1000, "steer": 20},
                    {"l": 1000, "steer": 110},
                    {"l": 1000, "steer": 30},
                    {"l": 1000, "steer": 90},
                    {"l": 1000, "steer": 0},
                    {"l": 1000, "steer": 90},
                    {"l": 1000, "steer": 150},
                    {"l": 1000, "steer": -210},
                    {"l": 1000, "steer": -155},
                    {"l": 1000, "steer": -130},
                    {"l": 1000, "steer": 0},
                    {"l": 1000, "steer": 90},
                    {"l": 1000, "steer": 180},
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
            # n_lm = secrets.choice(range(0,200))  # numero di landmark da posizionare 
            n_lm = secrets.choice(range(0,10))  # numero di landmark da posizionare 
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
        
    
    #def distanza_cartesiana(p,angolo,lm):
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
        #rad = math.radians(angolo)
        
        # r_x = p['x'] * math.cos(rad) - p['y'] * math.sin(rad)
        # r_y = p['x'] * math.sin(rad) + p['y'] * math.cos(rad)
        # x = lm['x'] - r_x  
        # y = lm['y'] - r_y  
        
        # 210722 skype con luigi
        d_x = lm['x'] - p['x']
        d_y = lm['y'] - p['y']
        x = d_x * math.cos(rad) + d_y * math.sin(rad)
        y = - d_x * math.sin(rad) + d_y * math.cos(rad)
        
       
        dist = {'x':x, 'y': y}
        return dist
    
    
    def landmark_is_in_tracking(t,album_misure,idd):
        '''
        Qui c'è la regola usata per determinare se il landmark è in tracking

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        album_misure : TYPE
            DESCRIPTION.
        idd : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        '''
        if (t > 0 and (idd in album_misure[t-1])) or ( t == 0):
            return True
        else:
            return False
        
    
    def get_measure(t, album_misure):
        """
        0. Apre il file percorso.dat e prende la posizione indicizzata dal parametro 't'
        1. prendi tutti i punti vicini alla posizione corrente
        2. partendo dalla posizione (veritiera, presa da percorso.dat), per ogni punto calcola la distanza x-y 
        3. aggiorna album_misure[t] in modo che si sappia quali punti sono rientrati in questo istante nella misura

        Parameters
        ----------
        t : int
            indice per l'accesso casuale al file percorso.dat

        Returns
        -------
        None.

        """
        album_misure.append([])
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
        
        _vect = []
        _nuovi = []
        MAX_RAGGIO = 50.
        MAX_ANG = 20.
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
                    album_misure[t].append(lm['idd'])                      
                    distanza = TestBed.distanza_cartesiana(p,radians,lm)  
                    if (TestBed.landmark_is_in_tracking(t,album_misure,lm['idd'])):  # Se è in tracking 
                        _vect.append(lm['idd'])
                        _vect.append(distanza['x'])
                        _vect.append(distanza['y'])
                    else:  # non è in tracking quindi è nuovo
                       _nuovi.append(lm['idd'])
                       _nuovi.append(distanza['x'])
                       _nuovi.append(distanza['y'])
        file_lm.close()
        n = int(len(_vect)/3)
        vect = np.ones( (n*2, 1), dtype=float, order='F') 
        vect.flags.writeable = True
        for i in range(n):
            vect[i*2,0] = _vect[i*3 + 1]
            vect[i*2 + 1,0] = _vect[i*3 + 2]
        
        return _vect,vect,_nuovi
    
        
def main():
    TestBed.fai_percorso()
    TestBed.posiziona_land_marks()
    
    
    print("finito, pace e bene.")        

if __name__ == "__main__":
    main()
        
