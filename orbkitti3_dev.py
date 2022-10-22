#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 08:57:37 2019

@author: fgr81
"""
import numpy as np
import math
import cv2
import pykitti
import sys
import os 
from os import listdir
from os.path import isfile, join
import ekfaus as ekf
import tkinter as tki
from tkinter.ttk import *
from PIL import Image, ImageTk
import threading
from datetime import datetime


'''
nonLinH(state):
    evolve la parte dello stato che coincide con la misura ( cioè: i lm dello stato che sono nella misura)
    per cui bisogna sapere ( tramite idd) quali lm di state evolvere
    lo scopo è quello di confrontare la misura con il valore evoluto dello stato

'''
class Landmark:

    def __init__(self,
                 l_x = 0,
                 l_y = 0,
                 l_desc = 0,
                 r_x = 0,
                 r_y = 0,
                 r_desc = 0,
                 disp = -1,
                 t = -1,
                 Q = 0,
                 idd = -1):
        self.left = {'x': l_x, 'y': l_y}
        self.right = {'x': r_x, 'y': r_y}
        self.l_desc = l_desc
        self.r_desc = r_desc
        #self.point = { 'x': p_x, 'y': p_y, 'z': p_z}
        self.point = self.setPos3D(Q)
        self.assimilated_position = {'x' : 0., 'y' : 0., 'z' : 0.}
        self.disparity = disp
        self.delta_x = l_x - r_x
        self.cluster_id = -1
        self.idd = idd
        self.t = t
        self.track = -1
        self.XaRow_0 = []
        self.XaRow_1 = []
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

    def __str__(self):
        #t idd L_x L_y R_x R_y disparity vP[0] vP[1] vP[2]
        strr = str(self.t)
        strr += ' ' + str(self.idd)
        strr += ' ' + str(self.left['x'])
        strr += ' ' + str(self.left['y'])
        strr += ' ' + str(self.right['x'])
        strr += ' ' + str(self.right['y'])
        strr += ' ' + str(self.disparity)
        strr += ' ' + str(self.point['x'])
        strr += ' ' + str(self.point['y'])
        strr += ' ' + str(self.point['z'])
        strr += '\n'
        return strr

    def setIdd( self,  lms_f_prec):
        if len(lms_f_prec) == 0:
            self.idd = -1
            return -1,-1
        desc_f_prec = np.zeros((len(lms_f_prec), 32), dtype=np.uint8)
        for i in range(0,len(lms_f_prec)):
            desc_f_prec[i] = lms_f_prec[i].get_l_desc()
        query_desc = np.zeros((1,32), dtype=np.uint8)
        query_desc[0] = self.l_desc
        mts = self.bf.match(query_desc, desc_f_prec)
        mts_s = sorted(mts, key = lambda x:x.distance)
        if len(mts_s) > 0:
            pref = lms_f_prec[mts_s[0].trainIdx]
            threshold = 75
            if np.sqrt(np.square(self.left['x'] - pref.left['x']) + np.square(self.left['y'] - pref.left['y'])) < threshold:
                self.idd = pref.idd
                self.assimilated_position = pref.assimilated_position
                self.track = 1                
                self.XaRow_0 = pref.XaRow_0
                self.XaRow_1 = pref.XaRow_1
            else:
                self.idd = -1
        else:
            self.idd = -1
        return self.idd, mts_s[0].trainIdx

    def setPos3D(self, Q):
        x_l = self.left['x']
        y_l = self.left['y']
        x_r = self.right['x']
        y_r = self.right['y']
        self.disparity = math.sqrt(((x_l - x_r)**2) + ((y_l - y_r)**2))
        p = np.array([[x_l], [y_r], [self.disparity], [1.0]])
        pos3D_ = np.dot(Q,p)
        pos3D = np.array([pos3D_[0][0]/pos3D_[3][0], pos3D_[1][0]/pos3D_[3][0], pos3D_[2][0]/pos3D_[3][0] ] )
        #arctan = np.arctan(pos3D[2]/pos3D[0])
        point = {}
        point['x'] = pos3D[0]
        point['y'] = pos3D[1]
        point['z'] = pos3D[2]
        return point


    def get_idd( self):
        return self.idd
    def get_l_desc(self):
        return self.l_desc

'''

+--------------------------------------------------------------------------------------+
| Frame                                                                                |
+--------------------------------------------------------------------------------------+
| cv2.imread  img_L # immagine sx                                                      |
| cv2.imread  img_R # immagine dx                                                      |
| int  i_frame # indice del frame, è usato per ottenere i nomifile delle immagini      |
| []  lms # lista dei lm                                                               |
| int  last_idd                                                                        |   
| int  n_tracked  numero di lm nel frame che sono in tracking                          |
| []   idd_in_analysis                                                                 |  
+--------------------------------------------------------------------------------------+
| init()                                                                               |
+--------------------------------------------------------------------------------------+

'''
class Frame:

    def getMeasure(self):        
        #measure = np.zeros((c*2,1), dtype=float, order='F')
        measure = np.zeros((self.n_tracked*2,1), dtype=float, order='F')
        c = 0
        for lm in self.lms:
            if lm.track == 1:
                measure[c,0] = lm.point['z']
                c += 1 
                measure[c,0] = lm.point['x']
                c += 1
        return measure
    
    def setAssimilated(self, analysis, measure, Xa, sigma_estimate):
        c = 0
        ii = 0
        nc = Xa.shape[1]
        for lm in self.lms:
            if lm.track == 1 :
                for i in range(len(self.idd_in_analysis)):     # vado a prendere in analysis la posizione assimilata
                    if self.idd_in_analysis[i] == lm.idd:
                        lm.assimilated_position['z'] = analysis[3 + i*2]
                        lm.assimilated_position['x'] = analysis[3 + i*2+1]                
                        lm.XaRow_0[:] = Xa[3 + i*2,:]
                        lm.XaRow_1[:] = Xa[4 + i*2,:]
                        break
            else:    # prendo come posizione assimilata quella misurata
                lm.assimilated_position['z'] = analysis[0,0] + measure[ii*2,0] * math.cos(analysis[2,0]) + measure[ii*2+1,0] * math.sin(analysis[2,0])
                lm.assimilated_position['x'] = analysis[1,0] + measure[ii*2,0] * math.sin(analysis[2,0]) - measure[ii*2+1,0] * math.cos(analysis[2,0])
                ii += 1 
                lm.XaRow_0 = []
                lm.XaRow_1 = []
                for c in range(nc):
                    lm.XaRow_0.append(sigma_estimate)
                    lm.XaRow_1.append(sigma_estimate)
                  
        
    
    def nonLinH(self, aanalysis):
        global nonlinh_
        i = 0
        for lm in self.lms:
            idd_ = lm.idd
            for ii in range(len(self.idd_in_analysis)):
                if self.idd_in_analysis[ii] == idd_ :
                    #print(idd_)
                    dx = aanalysis[ (ii * 2) + 3,0] - aanalysis[0,0]
                    dy = aanalysis[ (ii * 2) + 4,0] - aanalysis[1,0]
                    d = math.sqrt(dx*dx + dy*dy)
                    angle = math.atan2(dy,dx) - aanalysis[2,0]
                    #print("check 3 linee")
                    #print(str(nonlinh_[i*2,0]))
                    #print(str(d * math.cos(angle)))
                    nonlinh_[i*2,0] = d * math.cos(angle)
                    #print(str(nonlinh_[i*2,0]))
                    nonlinh_[i*2+1,0] = -d * math.sin(angle)
                    i += 1
                #print("nonlinh_[ii*2,0]:" + str(nonlinh_[ii*2,0]) + " nonlinh_[ii*2+1,0]:" + str(nonlinh_[ii*2+1,0]))
        #print("\ndentro nonLinH, ecco  nonlinh_:\n" + str(nonlinh_) + str("\n"))
        
        return nonlinh_
        

    def createAnalysis(self, stato):
        analysis = np.zeros((len(self.lms)*2+5,1), dtype=float, order='F')
        analysis[0,0] = stato[0]
        analysis[1,0] = stato[1]
        analysis[2,0] = stato[2]
        analysis[-2,0] = stato[-2]
        analysis[-1,0] = stato[-1]
        c = 3
        self.idd_in_analysis = []
        for lm in self.lms:
            analysis[c,0] = lm.assimilated_position['z']
            c += 1
            analysis[c,0] = lm.assimilated_position['x']
            c += 1
            self.idd_in_analysis.append(lm.idd)
        return analysis
    
    def createXa(self, _xa):
        
        nc = _xa.shape[1]
        xa = np.zeros((len(self.lms)*2+5, nc), dtype=float, order='F')
        #xa = np.ones((len(self.lms)*2+5, nc), dtype=float, order='F')
        xa.flags.writeable = True
        xa[0,:] = _xa[0,:]
        xa[1,:] = _xa[1,:]
        xa[2,:] = _xa[2,:]
        xa[-2,:] = _xa[-2,:]
        xa[-1,:] = _xa[-1,:]
        c = 3
        for lm in self.lms:
            xa[c,:] = lm.XaRow_0
            c += 1            
            xa[c,:] = lm.XaRow_1
            c += 1    
        #print("dentro createXa, shape di Xa:{0}", format(str(xa.shape)))
        return xa
       
    def __init__(self, i_frame = "", orbkitti = None, NP = 200, MAX_DISPARITY = 100, lms_f_prec = [], last_idd=0, idd_in_analysis = []):
        self.n_tracked = 0
        self.i_frame = i_frame
        self.last_idd = last_idd
        self.idd_in_analysis = idd_in_analysis
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
        lms = []
        lms2 = []
        lms3 = []
        delta_x_c =  {}
        delta_x_c[0] = []
        delta_x_c[1] = []
        delta_x_c[2] = []
        q1 = [0,0,0]
        q3 = [0,0,0]
        iqr = [0,0,0]
        lower_bound = [0,0,0]
        upper_bound = [0,0,0]
        f = i_frame.zfill(10) + '.png'
        img_L_ = cv2.imread(orbkitti.perc00 + '/' + f, cv2.IMREAD_GRAYSCALE)
        img_R_ = cv2.imread(orbkitti.perc01 + '/' + f, cv2.IMREAD_GRAYSCALE)
        h1, w1 = img_L_.shape[:2]
        #img2_ = np.zeros((h1*2, w1), np.uint8)
        #img2_[:h1, :w1] = img_L
        #img2_[h1:h1*2, :w1] = img_R
        self.img_L = cv2.cvtColor(img_L_, cv2.COLOR_GRAY2RGB)  # kiki controllare se fare questa conversione o se tenere in gray
        self.img_R = cv2.cvtColor(img_R_, cv2.COLOR_GRAY2RGB)
        #img2 = cv2.cvtColor(img2_, cv2.COLOR_GRAY2RGB)
        kp_L, desc_L = orb.detectAndCompute(self.img_L, None)    # detectAndCompute fornisce X coppie  keypoint,descriptor
        kp_R, desc_R = orb.detectAndCompute(self.img_R, None)
        matches = bf.match(desc_L, desc_R)
        matches = sorted(matches, key = lambda x:x.distance)
        for m in matches[:NP*2]:
            pts_L = kp_L[m.queryIdx].pt
            pts_R = kp_R[m.trainIdx].pt
            # ***  Creo il landmark
            lm = Landmark( t = i_frame,
                          l_x = pts_L[0], l_y = pts_L[1],
                          r_x = pts_R[0], r_y = pts_R[1],
                          l_desc = desc_L[m.queryIdx], r_desc = desc_R[m.trainIdx], Q = orbkitti.Q)
            # *** E lo pongo nella lista lms
            if abs(lm.delta_x) > MAX_DISPARITY:
                continue  # butta il lm
            lms.append( lm )
            #####
            #
            # Associo il lm ad uno dei tre cluster: sinistra, centro, destra
            #
            #####
            if lm.left['x'] < w1/3:
                lm.cluster_id = 0
                delta_x_c[0].append(lm.delta_x)
            elif lm.left['x'] < w1*2/3:
                lm.cluster_id = 1
                delta_x_c[1].append(lm.delta_x)
            else:
                lm.cluster_id = 2
                delta_x_c[2].append(lm.delta_x)
        # end for m in matches[:NP*2]:
        #####
        #
        # Filtro i landmark che sono outlier
        # per cui prima calcolo i limiti
        # inferiore e superiore
        #
        #####
        '''mean_c[0] = np.mean(delta_x_c[0])
        mean_c[1] = np.mean(delta_x_c[1])
        mean_c[2] = np.mean(delta_x_c[2])
        std_c[0] = np.std(delta_x_c[0])
        std_c[1] = np.std(delta_x_c[1])
        std_c[2] = np.std(delta_x_c[2])'''
        delta_x_c[0] = sorted(delta_x_c[0])
        delta_x_c[1] = sorted(delta_x_c[1])
        delta_x_c[2] = sorted(delta_x_c[2])
        if len(delta_x_c[0]) > 0:
            q1[0], q3[0]= np.percentile(delta_x_c[0],[25,75])
        else:
            q1[0] = 0
            q3[0] = 0
        if len(delta_x_c[1]) > 0:
            q1[1], q3[1]= np.percentile(delta_x_c[1],[25,75])
        else:
            q1[1] = 0
            q3[1] = 0
        if len(delta_x_c[2]) > 0:
            q1[2], q3[2]= np.percentile(delta_x_c[2],[25,75])
        else:
            q1[2] = 0
            q3[2] = 0
        iqr[0] = q3[0] - q1[0]
        iqr[1] = q3[1] - q1[1]
        iqr[2] = q3[2] - q1[2]
        lower_bound[0] = q1[0] -(1.5 * iqr[0])
        lower_bound[1] = q1[1] -(1.5 * iqr[1])
        lower_bound[2] = q1[2] -(1.5 * iqr[2])
        upper_bound[0] = q3[0] +(1.5 * iqr[0])
        upper_bound[1] = q3[1] +(1.5 * iqr[1])
        upper_bound[2] = q3[2] +(1.5 * iqr[2])
        for lm in lms:
            if lm.delta_x > lower_bound[lm.cluster_id] and lm.delta_x < upper_bound[lm.cluster_id]:
                lms2.append(lm)
        # end filtro-outlier
        #####
        #
        #  Controllo sul numero di lm per cluster:
        #  potrebbe capitare che un cluster laterale sia vuoto rispetto all'opposto
        #  a causa di differenze di luce, in particolare se c'è ombra su un lato
        #
        #####
        left = 0
        p_left = 0
        center = 0
        p_center = 0
        right = 0
        p_right = 0
        for lm in lms2:
            if lm.cluster_id == 0:
                left += 1
            elif lm.cluster_id == 1 :
                center += 1
            else:
                right += 1
        # devo decidere quanti lm per cluster prendere ( p_left, p_center, p_right)
        if left == 0:
            p_left = 0
            if center > 0:
                p_right = 0
                p_center = np.min((NP,center))
            else:  # center == 0
                p_center = 0
                p_right = np.min((NP,right))
        elif right == 0:
            p_right = 0
            if center > 0:
                p_left = 0
                p_center = np.min((NP,center))
            else:
                p_center = 0
                p_left = np.min((NP,left))
        else:
            if center < NP:
                p_center = center
                mmm = int((NP-p_center)/2)
                p_right = np.min((right,left,mmm))
                #p_left = NP - p_center - p_right
                p_left = p_right
            else:
                p_center = NP
                p_right = 0
                p_left = 0
        # filtro sulla base del numero di lm per cluster appena deciso
        t = 0
        while (p_left > 0 or p_center > 0 or p_right > 0) and ( t < len(lms2)):
            if lms2[t].cluster_id == 0:
                if p_left > 0:
                    p_left -= 1
                    lms3.append(lms2[t])
            elif lms2[t].cluster_id == 1:
                if p_center > 0:
                    p_center -= 1
                    lms3.append(lms2[t])
            else:
                if p_right > 0:
                    p_right -= 1
                    lms3.append(lms2[t])
            t += 1
        #end while (p_left > 0 or p_center > 0 or p_right > 0) and ( t < len(lms2)):
        #####
        #
        #  Ora devo capire se i match sono in ***TRACKING*** oppure no,
        #  cioè cercare nei landmark del frame precendente un eventuale corrispondente          
        #
        #####
        if len(lms_f_prec) == 0:  # primo frame
            for lm in lms3:
                lm.idd = self.last_idd
                self.last_idd += 1
        else:
            for lm in lms3:
                n_idd, i_lm_to_erase = lm.setIdd(lms_f_prec)
                if  n_idd == -1:
                    lm.idd = self.last_idd
                    self.last_idd += 1
                else:
                    self.n_tracked += 1
                    # Rimuovo da lms_f_prec il landmark usato per il track.,
                    # in modo da non riutilizzarlo
                    del lms_f_prec[i_lm_to_erase]
        # Ordino lms3 sulla base di idd
        sorted_lms3 = sorted(lms3, key = lambda k: k.get_idd())
        self.lms = sorted_lms3

'''
class State:
    def __init__(self,
                 x = 0.0,
                 y = 0.0,
                 phi = 0.0,
                 v = 0.0,
                 g = 0.0):
        self.x = x
        self.y = y
        self.phi = phi
        self.v = v
        self.g = g
  '''      


class Orbkitti:
    
    """Appling ORB to detect and match feature over KITTI images.
    
    Parameters
    ------------
    basedir : string
    basedir2: Images path
    date : string
    drive : string
    
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications in every epoch.
    """

    def __init__(self,
                 basedir = '/home/fgr81/kitti',
                 basedir_out = '/home/fgr81/Desktop/cantiere_EKF_AUS/SOFTX-D-17-00068-master/PalatellaGrassoSoftwareEKF-AUS-NL_SoftwareX_Reubmitted28nov2017',
                 date = '2011_10_03',
                 drive = '0027',
                 log = 'log.txt'):
        self.basedir = basedir
        self.basedir2 = basedir_out
        self.date = date
        self.drive = drive
        self.NF = 10000   # num files to open # todo implementare
        self.data = pykitti.raw(basedir, date, drive,frames=range(0,20,1), imformat = 'cv2')
        self.perc00 = basedir + '/' + date + '/' + date + '_drive_' + drive + '_sync/image_00/data'
        self.perc01 = basedir + '/' + date + '/' + date + '_drive_' + drive + '_sync/image_01/data'
        self.perc02 = basedir + '/' + date + '/' + date + '_drive_' + drive + '_sync/image_02/data'
        self.perc03 = basedir + '/' + date + '/' + date + '_drive_' + drive + '_sync/image_03/data'
        self.onlyfiles00 = [f for f in listdir(self.perc00) if isfile(join(self.perc00, f))]
        self.log = log
        self.data = pykitti.raw(basedir, date, drive,frames=range(0,20,1), imformat = 'cv2')
        self.calib = self.data._load_calib_cam_to_cam('calib_velo_to_cam.txt', 'calib_cam_to_cam.txt')
        self.Cx = self.calib['K_cam0'][0][2]
        self.Cy = self.calib['K_cam0'][1][2]
        self.f = self.calib['K_cam0'][0][0]
        self.Tx = self.calib['P_rect_10'][0][3] / self.f
        self.fuoco = self.f
        # print('Parametri di calibrazione: --- Cx= ' + str(Cx) + ' Cy= ' + str(Cy) + ' f= ' + str(f) + ' Tx= ' + str(Tx))
        """
        points = cv.reprojectImageTo3D(disp, Q)
        The form of the Q matrix is given as follows:
        """
        self.Q = np.array([[1, 0, 0, -self.Cx],
               [0, 1, 0, -self.Cy],
               [0, 0, 0, self.f],
               [0, 0, -1/self.Tx, 0]])

    
    def make_matches(self,
                     steering = True,
                     NP = 200,    # numero max di feature per frame
                     SOGLIA_DIST = 1.,
                     MAX_DISTANZA = 300.0,
                     MIN_DISTANZA = 1.,
                     MAX_DISPARITY = 100,
                     MAX_DISPARITY_TEMPORALE = 75,
                     MAX_ALTEZZA = -999.5,  # Attenzione: le cam sono montate ad una certa altezza ( quale ?), inoltre al crescere dell'altezza la y diminuisce !
                     BANDA_LATERALE_OSCURA = 40,
                     do_log = True,
                     vis_matches = False
                     ):

        NOMEFILE_M = 'matches_' + self.date + '_' + self.drive + '.dat'
        #NOMEFILE_S = 'steering_' + self.date + '_' + self.drive + '.dat'
        NOMEFILELOG = self.log
        file = open(self.basedir2 + '/' + NOMEFILE_M, 'w')
        file.write('t idd L_x L_y R_x R_y disparity vP[0] vP[1] vP[2]\n')  # intestazione

        NOMEFILELOG = self.log
        if do_log:
            flog = open(NOMEFILELOG, 'w')

        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

        lms = []
        lms2 = []
        lms3 = []
        lms_f_prec = []
        #desc_f_prec = []
        delta_x_c =  {}
        delta_x_c[0] = []
        delta_x_c[1] = []
        delta_x_c[2] = []

        i_frame = 0
        last_idd = 0

        ff = sorted(self.onlyfiles00)

        for f in ff:
            lms.clear()
            lms2.clear()
            lms3.clear()
            delta_x_c[0].clear()
            delta_x_c[1].clear()
            delta_x_c[2].clear()

            mean_c = [0,0,0]
            std_c = [0,0,0]
            q1 = [0,0,0]
            q3 = [0,0,0]
            iqr = [0,0,0]
            lower_bound = [0,0,0]
            upper_bound = [0,0,0]

            #landmarks.append([])
            img_L = cv2.imread(self.perc00 + '/' + f, cv2.IMREAD_GRAYSCALE)
            img_R = cv2.imread(self.perc01 + '/' + f, cv2.IMREAD_GRAYSCALE)

            h1, w1 = img_L.shape[:2]
            img2_ = np.zeros((h1*2, w1), np.uint8)
            img2_[:h1, :w1] = img_L
            img2_[h1:h1*2, :w1] = img_R
            img2 = cv2.cvtColor(img2_, cv2.COLOR_GRAY2RGB)

            kp_L, desc_L = orb.detectAndCompute(img_L, None)    # detectAndCompute fornisce X coppie  keypoint,descriptor
            kp_R, desc_R = orb.detectAndCompute(img_R, None)
            matches = bf.match(desc_L, desc_R)
            matches = sorted(matches, key = lambda x:x.distance)
            #radius = []

            for m in matches[:NP*2]:
                pts_L = kp_L[m.queryIdx].pt
                pts_R = kp_R[m.trainIdx].pt

                # ***  Creo il landmark
                lm = Landmark( t = i_frame,
                              l_x = pts_L[0], l_y = pts_L[1],
                              r_x = pts_R[0], r_y = pts_R[1],
                              l_desc = desc_L[m.queryIdx], r_desc = desc_R[m.trainIdx])
                # *** E lo pongo nella lista lms
                if abs(lm.delta_x) > MAX_DISPARITY:
                    continue  # butta il lm
                lms.append( lm )
                #####
                #
                # Associo il lm ad uno dei tre cluster: sinistra, centro, destra
                #
                #####
                if lm.left['x'] < w1/3:
                    lm.cluster_id = 0
                    delta_x_c[0].append(lm.delta_x)
                elif lm.left['x'] < w1*2/3:
                    lm.cluster_id = 1
                    delta_x_c[1].append(lm.delta_x)
                else:
                    lm.cluster_id = 2
                    delta_x_c[2].append(lm.delta_x)
            # end for m in matches[:NP*2]:

            #####
            #
            # Filtro i landmark che sono outlier
            # per cui prima calcolo i limiti
            # inferiore e superiore
            #
            #####
            '''mean_c[0] = np.mean(delta_x_c[0])
            mean_c[1] = np.mean(delta_x_c[1])
            mean_c[2] = np.mean(delta_x_c[2])
            std_c[0] = np.std(delta_x_c[0])
            std_c[1] = np.std(delta_x_c[1])
            std_c[2] = np.std(delta_x_c[2])'''
            delta_x_c[0] = sorted(delta_x_c[0])
            delta_x_c[1] = sorted(delta_x_c[1])
            delta_x_c[2] = sorted(delta_x_c[2])
            if len(delta_x_c[0]) > 0:
                q1[0], q3[0]= np.percentile(delta_x_c[0],[25,75])
            else:
                q1[0] = 0
                q3[0] = 0
            if len(delta_x_c[1]) > 0:
                q1[1], q3[1]= np.percentile(delta_x_c[1],[25,75])
            else:
                q1[1] = 0
                q3[1] = 0
            if len(delta_x_c[2]) > 0:
                q1[2], q3[2]= np.percentile(delta_x_c[2],[25,75])
            else:
                q1[2] = 0
                q3[2] = 0
            iqr[0] = q3[0] - q1[0]
            iqr[1] = q3[1] - q1[1]
            iqr[2] = q3[2] - q1[2]
            lower_bound[0] = q1[0] -(1.5 * iqr[0])
            lower_bound[1] = q1[1] -(1.5 * iqr[1])
            lower_bound[2] = q1[2] -(1.5 * iqr[2])
            upper_bound[0] = q3[0] +(1.5 * iqr[0])
            upper_bound[1] = q3[1] +(1.5 * iqr[1])
            upper_bound[2] = q3[2] +(1.5 * iqr[2])
            for lm in lms:
                if lm.delta_x > lower_bound[lm.cluster_id] and lm.delta_x < upper_bound[lm.cluster_id]:
                    lms2.append(lm)
            # end filtro-outlier


            #####
            #
            #  Controllo sul numero di lm per cluster:
            #  potrebbe capitare che un cluster laterale sia vuoto rispetto all'opposto
            #  a causa di differenze di luce, in particolare se c'è ombra su un lato
            #
            #####
            left = 0
            p_left = 0
            center = 0
            p_center = 0
            right = 0
            p_right = 0
            for lm in lms2:
                if lm.cluster_id == 0:
                    left += 1
                elif lm.cluster_id == 1 :
                    center += 1
                else:
                    right += 1
            # devo decidere quanti lm per cluster prendere ( p_left, p_center, p_right)
            if left == 0:
                p_left = 0
                if center > 0:
                    p_right = 0
                    p_center = np.min((NP,center))
                else:  # center == 0
                    p_center = 0
                    p_right = np.min((NP,right))
            elif right == 0:
                p_right = 0
                if center > 0:
                    p_left = 0
                    p_center = np.min((NP,center))
                else:
                    p_center = 0
                    p_left = np.min((NP,left))
            else:
                if center < NP:
                    p_center = center
                    mmm = int((NP-p_center)/2)
                    p_right = np.min((right,left,mmm))
                    #p_left = NP - p_center - p_right
                    p_left = p_right
                else:
                    p_center = NP
                    p_right = 0
                    p_left = 0
            print('left center right:' + str(left) + ' ' + str(center) + ' ' + str(right) + '\n')
            print('p_left p_center p_right:' + str(p_left) + ' ' + str(p_center) + ' ' + str(p_right) + '\n')
            if do_log:
                flog.write('p_left p_center p_right:' + str(p_left) + ' ' + str(p_center) + ' ' + str(p_right) + '\n')
            # filtro sulla base del numero di lm per cluster appena deciso
            t = 0
            while (p_left > 0 or p_center > 0 or p_right > 0) and ( t < len(lms2)):
                if lms2[t].cluster_id == 0:
                    if p_left > 0:
                        p_left -= 1
                        lms3.append(lms2[t])
                elif lms2[t].cluster_id == 1:
                    if p_center > 0:
                        p_center -= 1
                        lms3.append(lms2[t])
                else:
                    if p_right > 0:
                        p_right -= 1
                        lms3.append(lms2[t])
                t += 1


            ''' ora devo capire se i match sono in tracking oppure no,
            cioè cercare nei landmark del frame precendente un eventuale corrispondente          '''

            if len(lms_f_prec) == 0:  # primo frame
                for lm in lms3:
                    lm.idd = last_idd
                    last_idd += 1
            else:
                for lm in lms3:
                    n_idd, i_lm_to_erase = lm.setIdd(lms_f_prec)
                    if  n_idd == -1:
                        lm.idd = last_idd
                        last_idd += 1
                    else:
                        # Rimuovo da lms_f_prec il landmark usato per il track.,
                        # in modo da non riutilizzarlo
                        del lms_f_prec[i_lm_to_erase]


            # Ordino lms3 sulla base di idd
            sorted_lms3 = sorted(lms3, key = lambda k: k.get_idd())


            #####
            # carico lms_f_prec per la prossima iterazione
            #####
            lms_f_prec.clear()
            #desc_f_prec.clear()
            for lm in sorted_lms3:
                lms_f_prec.append(lm)
                #desc_f_prec.append(lm.l_desc)

            # OUTPUT
            n_lm_in_track = 0
            for lm in sorted_lms3:
                if lm.track == 1:
                    n_lm_in_track += 1
                lm.setPos3D(self.Q)
                file.write(str(lm))

                if vis_matches:
                    if lm.track == 1:
                        color = (255,0,0)
                    else:
                        color = (0,255,0)
                    cv2.circle(img2,(int(lm.left['x']),int(lm.left['y'])), 4, color, 1)
                    cv2.circle(img2,(int(lm.right['x']), int(lm.right['y'])+h1)   , 4, color, 1)
                    cv2.line(img2, (int(lm.left['x']),int(lm.left['y']) ),
                             ( int(lm.right['x']), int(lm.right['y'])+h1 ), color,1)

            if do_log:
                log_s = str(i_frame) + ' ***  #lms: ' + str(len(lms)) + ' #lms2: ' + str(len(lms2)) + ' #lms3: ' + str(len(sorted_lms3)) + ' #lm_in_track:' + str(n_lm_in_track) + '\n'
                flog.write(log_s)
                print(log_s)
                for lm in sorted_lms3:
                    flog.write(str(lm.idd) + '\n')

            if vis_matches:
                cv2.imshow(f,img2)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            i_frame += 1


        if do_log:
            flog.close()
        file.close()
        print('finito, pace e bene.')
        ##### end  make_matches()





class App:
    def __init__(self, primo, ultimo):
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.root = tki.Tk()
        self.panel = None
        btn = tki.Button(self.root, text="Daje con lo START !",    command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,    pady=10)
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=(primo, ultimo))
        self.thread.start()
        self.root.wm_title("Forza FAbio")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        self.root.destroy()
        self.root.quit()
        
    def takeSnapshot(self):
            ts = datetime.datetime.now()
            filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            cv2.imwrite(filename, self.frame.copy())
            print("[INFO] saved {}".format(filename))

    '''
N0 : X
     Y
     Phi
     V
     G
fp: measOperator
np: gmunu(N0 + landmarkperframe,1) -> devo inizializzare con gmunu_estimate = 1.
np: ModelError(2,1)
np: analysis(N0 + landmarkperframe,1) -> si inizializza con sigma_estimate = 1.
np: measure(landmarkperframe*2+1,1)


    
    velocità iniziale
    sterzata iniziale

    '''
    def videoLoop(self, primo, ultimo):
        try:
            last_idd = 0
            lms_f_prec = []  # serve per trovare il lm nel frame precedente
            t = datetime.now().microsecond

            for i in range (primo, ultimo):
                #print(i)                  
                lms_f_prec.clear()
            
                if i > primo:  # siamo in tracking
                    for lm in self.frame.lms:  # carico lms_f_prec 
                        lms_f_prec.append(lm)     
                        
                self.frame = Frame(i_frame=str(i), orbkitti=kk, lms_f_prec=lms_f_prec, last_idd=last_idd)
                last_idd = self.frame.last_idd
                
                if i == primo:
                        h1, w1 = self.frame.img_L.shape[:2]
                        img2 = np.zeros((h1*2, w1, 3), np.uint8)
                img2[:h1, :w1] = self.frame.img_L
                img2[h1:h1*2, :w1] = self.frame.img_R
                for lm in self.frame.lms:
                    if lm.track == 1:
                        color = (0,255,0)  
                    else:
                        color = (255,0,0)
                    cv2.circle(img2,(int(lm.left['x']),int(lm.left['y'])), 4, color, 1)
                    cv2.circle(img2,(int(lm.right['x']), int(lm.right['y'])+h1)   , 4, color, 1)
                    cv2.line(img2, (int(lm.left['x']),int(lm.left['y']) ), ( int(lm.right['x']), int(lm.right['y'])+h1 ), color,1)
                
                m = datetime.now().microsecond                
                mm = m
                if m < t:                    
                    m += 1000000                    
                fps = 1/((m - t) * 0.000001)

                t = mm
                
                log_s = str(i) + ' fps:' + "{0:.2f}".format(fps) + ' ***  #lms: ' + str(len(self.frame.lms)) 
                cv2.putText(img2, log_s, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 2)
                                
                image1 = img2
                image1 = Image.fromarray(image1)
                image1 = ImageTk.PhotoImage(image1)
                if self.panel is None:
                    self.panel = tki.Label(image=image1)
                    self.panel.image = image1
                    self.panel.pack(side="left", padx=10, pady=10)
                else:
                    self.panel.configure(image=image1)
                    self.panel.image = image1
        except:
            print("[INFO] caught a RuntimeError", sys.exc_info()[0], "     ", sys.exc_info()[1] )




def readFile(forecast):
    
    f = open(forecast, 'r')
    x = float(f.readline())
    y = float(f.readline())
    phi = float(f.readline())
    V = float(f.readline())
    G = float(f.readline())
    N = int(f.readline())
    xf = np.empty(N*2 + 5, dtype=float)
    xf[0] = x
    xf[1] = y
    xf[2] = phi
    for i in range(0,N):
        linea = f.readline()
        xn, yn = [float(s) for s in linea.split()]
        xf[ 3 + 2 * i] = xn
        xf[ 4 + 2 * i] = yn
    xf[ 2 * N + 3] = V
    xf[ 2 * N + 4] = G
    f.close()
    return xf
    
def writeFile(forecast, xf):
    f = open(forecast,'w')
    f.write(str(xf[0]) + '\n')
    f.write(str(xf[1]) + '\n')
    f.write(str(xf[2]) + '\n')
    f.write(str(xf[-2]) + '\n')
    f.write(str(xf[-1]) + '\n')
    N = int((xf.shape[0] - 5) / 2)
    f.write(str(N) + '\n')
    for i in range(0,N):
        f.write(str(xf[3 + 2*i]) + ' ' + str(xf[4 + 2 * i]) + '\n')
    f.close()
    


def evolve(XX, time):
  nsteps = int(time / 0.025)
  comando = "./MySlam statotemporaneo.dat " + str(nsteps)  
  for k in range(0,XX.shape[1]):      
      writeFile("statotemporaneo.dat", XX[:,k])
      os.system(comando)
      colonna = readFile("statotemporaneo.dat")
      for i in range(0,colonna.shape[0]):
          XX[i,k] = colonna[i]


#def NonLinH(state):    
#    global nonlinh_
'''
    dx = state( (ii * 2) + 3,0) - state(0,0);
        dy = state( (ii * 2) + 4,0) - state(1,0);
        d = sqrt(dx*dx + dy*dy);
        angle = atan2(dy,dx) - state(2,0);
        mis(kk*2,0) = d * cos(angle);
        mis(kk*2+1,0) = -d * sin(angle);      //LP, il - ci va perché per kitti la x è a destra
'''    
 #   return nonlinh_

# ppi = 6.28318530717959
ppi = 2*math.pi
kitti = Orbkitti()
#kk.make_matches(vis_matches=False,do_log=True)

primo = 0
#ultimo = 4543
ultimo = 2

os.system('rm statotemporaneo.dat')
f = open('traiettorie.dat','w')
N0 = 5  #  X Y Phi V G
sigmad = 5
sigmaA = 3 * ppi / 360.  #  1 grado sessagesimale 
sigma_estimate = 1.  # nei componenti della matrice Xa, è la stima iniziale della sigma
gmunu_estimate = 1.  #  valore iniziale delle componenti di gmunu
velocita_iniziale = 0. 
sterzata_iniziale = 0.0;
#####
#
#  Initializing the EKF_AUS_NL class with 
#  N0 degrees of freedom,
#  6 linear Lyapunov vectors, N0-2 number of measurements,
#  1 nonlinear interaction
#
#####
ekf = ekf.EkfAus(N0, 6, 0, 1)
ekf.AddInflation(1.e-1)
ekf.LFactor(0.01)
nc = ekf.TotNumberPert()

last_idd = 0

new_frame = Frame(i_frame=str(primo), orbkitti=kitti, lms_f_prec=[], last_idd=last_idd)
nlm = len(new_frame.lms)
#measure = np.zeros((nlm*2,1), dtype=float, order='F')
#c = 0
#for lm in new_frame.lms:
#    measure[c,0] = lm.point['z']
#    c += 1   
#    measure[c,0] = lm.point['x']
#    c += 1
Xa = np.zeros((N0 + nlm*2, nc), dtype=float, order='F')
Xa.flags.writeable = True
Xa[0:3,:] = 0.
Xa[3:,:] = sigma_estimate 
Xa[-2,:] = np.random.random_sample((nc,))
Xa[-1,:] = np.random.random_sample((nc,)) 

analysis = np.zeros((N0 + nlm*2,1), dtype=float, order='F')
for i in range(0,nlm):
    analysis[3 + i*2,0] = new_frame.lms[i].point['z']
    analysis[4 + i*2,0] = new_frame.lms[i].point['x']
    new_frame.idd_in_analysis.append(new_frame.lms[i].idd)
analysis[3 + nlm*2,0] = velocita_iniziale
analysis[4 + nlm*2,0] = sterzata_iniziale

gmunu = np.ones((N0 + nlm*2,1), dtype=float, order='F')
gmunu.flags.writeable = True
gmunu[:,:] = gmunu * gmunu_estimate

ekf.P( nlm * 2 )
R = np.ones((ekf.P(),1), dtype=float, order='F') * sigmad * sigmad
R[-1,0] = 1.e-4
ModelError = np.ones((2,1), dtype=float, order='F') 
ModelError.flags.writeable = True
ModelError[0,0] = 2. #  (m/s) error in the velocity  
ModelError[1,0] = 90*ppi/360. # 3 degrees error for the steering angle 

# da qui iniziano le operazioni nel ciclo delle assimilazioni
for i_frame in range(primo+1,ultimo):
    print("i_frame = {}".format(i_frame))
    ekf.SetModelErrorVariable(Xa.shape[0]-2, Xa.shape[0]-1,ModelError, Xa);

    ekf.N(N0 + nlm*2)  # IMPORTANTE ! 
    
    perturbazioni = ekf.PrepareForEvolution(analysis, Xa, gmunu)
    evolve(perturbazioni, 0.1);
    evolve(analysis, 0.1);
    Xa = ekf.PrepareForAnalysis(analysis, perturbazioni, gmunu);
    lms_f_prec = new_frame.lms
    new_frame = Frame(i_frame=str(i_frame), orbkitti=kitti, lms_f_prec= lms_f_prec,last_idd=new_frame.last_idd, idd_in_analysis = new_frame.idd_in_analysis)
    nlm = len(new_frame.lms)
    measure = new_frame.getMeasure()
    print("len measure:{0}".format(len(measure)/2))
    ekf.P(len(measure))
    #R = np.ones((ekf.P(),1), dtype=float, order='F') * sigmad * sigmad
    R = np.ones((len(measure),1), dtype=float, order='F') * sigmad * sigmad
    R[-1,0] = 0.1
    
    # E' importante dichiarare qui la variabile che sarà restituita da nonLinH()
    nonlinh_ = np.zeros(shape=measure.shape, dtype=float, order='F')
    nonlinh_.flags.writeable = True
    ekf.Assimilate(measure, new_frame.nonLinH, R, analysis, Xa);
    print( str(analysis[0,0]) + ' ' + str(analysis[1,0]) + ' ' + str(analysis[2,0]) + ' ' + str(analysis[-2,0]) + ' ' + str(analysis[-1,0]) + '\n')
    f.write( str(analysis[0,0]) + ' ' + str(analysis[1,0]) + ' ' + str(analysis[2,0]) + ' ' + str(analysis[-2,0]) + ' ' + str(analysis[-1,0]) + '\n')
    
    new_frame.setAssimilated(analysis, measure, Xa, sigma_estimate)  # aggiorno in Frame le posizioni dei lm sulla base dell'assimilazione
    analysis = new_frame.createAnalysis(analysis)   # rinnovo il vettore analysis con i nuovi lm, preparandolo per la prox iterazione 
    Xa = new_frame.createXa(Xa)
    gmunu = np.ones((len(new_frame.lms)*2+5,1),dtype=float, order='F') * gmunu_estimate
    
f.close()
#app = App(primo, ultimo)
#app.root.mainloop()
