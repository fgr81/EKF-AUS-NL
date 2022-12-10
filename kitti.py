#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:31:04 2022

@author: fgr81
"""
#####
#
# Programma con GUI 
#
#####

import PySimpleGUI as sg
import cv2
import pykitti
import numpy as np
import time
import os
from PIL import Image
import math
#import Slam
#from ekf_aus_utils import EkfAusUtils as Ekf



class OrbMark():
    
    def __init__(self,
                 l_x=0,
                 l_y=0,
                 l_desc=0,
                 r_x=0,
                 r_y=0,
                 r_desc=0,
                 t=-1,
                 idd=-1
                 ):
        self.left = {'x': l_x, 'y': l_y}
        self.right = {'x': r_x, 'y': r_y}
        self.l_desc = l_desc
        self.r_desc = r_desc
        self.delta_x = l_x - r_x
        self.cluster_id = -1
        self.t = t
        #self.id = OrbMark.last_id + 1
        self.idd = idd
        # print(f"- OrbMark.__init__self.id:{self.id}")
        #OrbMark.last_id += 1        

    def make_3d_point(self, parameter):
        # global parameters
        q = parameter.q
        x_l = self.left['x']
        y_l = self.left['y']
        x_r = self.right['x']
        y_r = self.right['y']
        disparity = math.sqrt(((x_l - x_r)**2) + ((y_l - y_r)**2))
        p = np.array([[x_l], [y_r], [disparity], [1.0]])
        pos3D_ = np.dot(q, p)
        if disparity == 0.:
            raise ValueError('diparity null')
            '''
            p =
            array([[598.39135742],
       [161.24317932],
       [  0.        ],
       [  1.        ]])
            '''
        pos3D = np.array([pos3D_[0][0]/pos3D_[3][0],
                          pos3D_[1][0]/pos3D_[3][0],
                          pos3D_[2][0]/pos3D_[3][0]])
        # todo gestione eccezione dividebyzero
        point = {}
        # point['x'] = pos3D[0]
        # point['y'] = pos3D[1]
        # point['z'] = pos3D[2]
        point['x'] = pos3D[2]
        point['y'] = - pos3D[0]
        return point

    @staticmethod
    def do_orb(i, parameters):    
        '''
        Apre due immagini e applica ORB in modo da determinare gli orb_mark

        Parameters
        ----------
        firs_img_path : string
        secondo_img_path : string

        Returns
        -------
        array of OrbMark

        '''
        first_img_path = parameters['sx_path']
        second_img_path = parameters['dx_path']
        i_frame = str(i)
        lms = []
        lms2 = []
        lms3 = []
        delta_x_c = {}
        delta_x_c[0] = []
        delta_x_c[1] = []
        delta_x_c[2] = []
        q1 = [0, 0, 0]
        q3 = [0, 0, 0]
        iqr = [0, 0, 0]
        lower_bound = [0, 0, 0]
        upper_bound = [0, 0, 0]
        NP = parameters['np']
        MAX_DISPARITY = parameters['max_disparity']
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        f = i_frame.zfill(10) + '.png'
        img_L_ = cv2.imread(first_img_path + '/' + f, cv2.IMREAD_GRAYSCALE)
        img_R_ = cv2.imread(second_img_path + '/' + f, cv2.IMREAD_GRAYSCALE)
        h1, w1 = img_L_.shape[:2]
        img_L = cv2.cvtColor(img_L_, cv2.COLOR_GRAY2RGB)
        img_R = cv2.cvtColor(img_R_, cv2.COLOR_GRAY2RGB)
        # detectAndCompute fornisce X coppie  keypoint,descriptor
        kp_L, desc_L = orb.detectAndCompute(img_L, None)
        kp_R, desc_R = orb.detectAndCompute(img_R, None)
        matches = bf.match(desc_L, desc_R)
        matches = sorted(matches, key=lambda x: x.distance)
        for m in matches[:NP*2]:
            pts_L = kp_L[m.queryIdx].pt
            pts_R = kp_R[m.trainIdx].pt
            # ***  Creo il landmark
            lm = OrbMark(t=i_frame,
                         l_x=pts_L[0], l_y=pts_L[1],
                         r_x=pts_R[0], r_y=pts_R[1],
                         l_desc=desc_L[m.queryIdx],
                         r_desc=desc_R[m.trainIdx]
                         )
            # *** E lo pongo nella lista lms
            if abs(lm.delta_x) > MAX_DISPARITY:
                continue  # butta il lm
            if lm.left['x'] == lm.right['x'] and lm.left['y'] == lm.right['y']:
                continue  # butta lm che hanno disparity == 0
            lms.append(lm)
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
        delta_x_c[0] = sorted(delta_x_c[0])
        delta_x_c[1] = sorted(delta_x_c[1])
        delta_x_c[2] = sorted(delta_x_c[2])
        if len(delta_x_c[0]) > 0:
            q1[0], q3[0] = np.percentile(delta_x_c[0], [25, 75])
        else:
            q1[0] = 0
            q3[0] = 0
        if len(delta_x_c[1]) > 0:
            q1[1], q3[1] = np.percentile(delta_x_c[1], [25, 75])
        else:
            q1[1] = 0
            q3[1] = 0
        if len(delta_x_c[2]) > 0:
            q1[2], q3[2] = np.percentile(delta_x_c[2], [25, 75])
        else:
            q1[2] = 0
            q3[2] = 0
        iqr[0] = q3[0] - q1[0]
        iqr[1] = q3[1] - q1[1]
        iqr[2] = q3[2] - q1[2]
        lower_bound[0] = q1[0] - (1.5 * iqr[0])
        lower_bound[1] = q1[1] - (1.5 * iqr[1])
        lower_bound[2] = q1[2] - (1.5 * iqr[2])
        upper_bound[0] = q3[0] + (1.5 * iqr[0])
        upper_bound[1] = q3[1] + (1.5 * iqr[1])
        upper_bound[2] = q3[2] + (1.5 * iqr[2])
        for lm in lms:
            if lm.delta_x > lower_bound[lm.cluster_id] and lm.delta_x < upper_bound[lm.cluster_id]:
                lms2.append(lm)
        # end filtro-outlier
        #####
        #
        #  Controllo sul numero di lm per cluster:
        #  potrebbe capitare che un cluster laterale sia vuoto rispetto
        #  all'opposto a causa di differenze di luce,
        #  in particolare se c'è ombra su un lato
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
            elif lm.cluster_id == 1:
                center += 1
            else:
                right += 1
        # devo decidere quanti lm per cluster prendere
        # ( p_left, p_center, p_right)
        if left == 0:
            p_left = 0
            if center > 0:
                p_right = 0
                p_center = np.min((NP, center))
            else:  # center == 0
                p_center = 0
                p_right = np.min((NP, right))
        elif right == 0:
            p_right = 0
            if center > 0:
                p_left = 0
                p_center = np.min((NP, center))
            else:
                p_center = 0
                p_left = np.min((NP, left))
        else:
            if center < NP:
                p_center = center
                mmm = int((NP-p_center)/2)
                p_right = np.min((right, left, mmm))
                # p_left = NP - p_center - p_right
                p_left = p_right
            else:
                p_center = NP
                p_right = 0
                p_left = 0
        # filtro sulla base del numero di lm per cluster appena deciso
        t = 0
        while (p_left > 0 or p_center > 0 or p_right > 0) and (t < len(lms2)):
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
        return lms3

    def match(self, lms_f_prec, parameters):
        # global parameters
        threshold_spazio = parameters['threshold_spazio']
        threshold_matcher = parameters['threshold_matcher']
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        desc_f_prec = np.zeros((len(lms_f_prec), 32), dtype=np.uint8)
        for i in range(0, len(lms_f_prec)):
            desc_f_prec[i] = lms_f_prec[i].l_desc
        query_desc = np.zeros((1, 32), dtype=np.uint8)
        query_desc[0] = self.l_desc
        matches = matcher.match(query_desc, desc_f_prec)
        # print(f"FOUND {len(matches)} matches")
        if len(matches) > 0:
            sorted_matches = sorted(matches, key=lambda x: x.distance)
            pref = lms_f_prec[sorted_matches[0].trainIdx]
            dist = np.sqrt(np.square(self.left['x'] - pref.left['x']) +
                           np.square(self.left['y'] - pref.left['y']))
            # print(f"distanza spaziale:{dist}  --
            # distanza match:{sorted_matches[0].distance}")
            if dist < threshold_spazio and sorted_matches[0].distance < threshold_matcher:
                # print(f"idd del landmark selezionato:
                # {lms_f_prec[sorted_matches[0].trainIdx].id}")
                return lms_f_prec[sorted_matches[0].trainIdx]
        return None

class Kitti:
    
    def __init__(self):
        # base_dir_out = '/home/fgr81/Desktop/cantiere_EKF_AUS_py'
        basedir = '/home/fgr81/kitti'
        date = '2011_10_03'
        drive = '0027'
        data = pykitti.raw(basedir, date, drive, frames = range(0, 20, 1), imformat = 'cv2')
        self.sx_path = basedir + '/' + date + '/' + date + '_drive_' + drive + '_sync/image_00/data'
        self.dx_path = basedir + '/' + date + '/' + date + '_drive_' + drive + '_sync/image_01/data'
        calib = data._load_calib_cam_to_cam('calib_velo_to_cam.txt', 'calib_cam_to_cam.txt')
        Cx = calib['K_cam0'][0][2]
        Cy = calib['K_cam0'][1][2]
        f = calib['K_cam0'][0][0]  # fuoco
        Tx = calib['P_rect_10'][0][3] / f
        q = np.array([[1, 0, 0, -Cx],
                           [0, 1, 0, -Cy],
                           [0, 0, 0, f],
                           [0, 0, -1/Tx, 0]])
        self.last_idd = 0
        self.orb_parameters= {
            'sx_path': self.sx_path,
            'dx_path': self.dx_path,
            'q': q,
            'np': 150,  # numero di OrbMark per frame
            'max_disparity': 100,
            'threshold_spazio': 75.0,
            'threshold_matcher': 30.0
            }
        self.lm_t_prec = []
        
        
    def disegna_gui(self):
        layout = [
            [sg.Text(f"{0}", key='frame_index', size=(30, 1)),
             sg.Button("Avanti"),
             sg.Button("Auto"),
             sg.Button("Stop"),
             sg.Button("Esci")
             ],
            [sg.Text("due istanti successivi", key='info', size=(150, 1))
             ],
            [sg.Image(filename="", key="IMAGE")]
            ]
        window = sg.Window("Demo", layout)
        return window
    
    def update_gui(self, index, window, gui_lm_track):
        f = str(index).zfill(10) + '.png'
        img_00 = cv2.imread(self.sx_path + '/' + f, cv2.IMREAD_GRAYSCALE)
        img_01 = cv2.imread(self.dx_path + '/' + f, cv2.IMREAD_GRAYSCALE)
        f_prec = str(index - 1).zfill(10) + '.png'
        img_10 = cv2.imread(self.sx_path + '/' + f_prec, cv2.IMREAD_GRAYSCALE)
        img_h, img_w = img_00.shape[:2]
        canvas = np.zeros((img_h * 2, img_w * 2), np.uint8)
        canvas[:img_h, :img_w] = img_00
        canvas[:img_h, img_w:img_w * 2] = img_01
        canvas[img_h:img_h * 2, :img_w] = img_10
        gnuplot_command = f"gnuplot -e \"set terminal png size {img_w},{img_h}; unset ytics; unset xtics;"
        gnuplot_command += " set output \'plot.png\'; plot \'trajectory.dat\' using 1:2 with line\""
        os.system(gnuplot_command)
        img_11 = cv2.imread("plot.png", cv2.IMREAD_GRAYSCALE)
        canvas[img_h:img_h * 2, img_w:img_w * 2] = img_11
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        #for om in measure[index].track:
        for om in gui_lm_track:
            cv2.line(canvas,
                      (int(om['om'].left['x']), int(om['om'].left['y'])),
                      (int(img_w + om['om'].right['x']), int(om['om'].right['y'])),
                      (0, 255, 0),
                      1)
            cv2.line(canvas,
                      (int(om['om'].left['x']), int(om['om'].left['y'])),
                      (int(om['om_p'].left['x']), img_h + int(om['om_p'].left['y'])),
                      (0, 0, 255),
                      1)
        canvas = cv2.resize(canvas, (img_w, img_h))
        img = Image.fromarray(canvas, 'RGB')
        img.save('tmp.png')
        ###
        # Aggiorno l'interfaccia
        ###
        window["IMAGE"].update(filename='tmp.png')
        # info = f"Lunghezza della misura:{len(measure[index].track)} Analysis: {analysis[0]} {analysis[1]} {analysis[2]} {analysis[-2]} {analysis[-1]}"
        info = "Ciaooooo"
        window["info"].update(info)
        window["frame_index"].update(f"frame index:{index}")\
    
    
    def get_scan(self, t):
        lm_t = OrbMark.do_orb(t, self.orb_parameters)
        gui_lm_track = []
        #####
        # Numero i lm e per farlo ho bisogno di conosce i lm del frame precedente
        #####
        for lm in lm_t:
            altrimenti = 0
            if len(self.lm_t_prec) > 0 or t > 0 :
                _lm_match = lm.match(self.lm_t_prec, self.orb_parameters)
                if _lm_match is not  None:
                    lm.id = _lm_match.id
                    gui_lm_track.append({
                        'om': lm,
                        'om_p': _lm_match
                        })
                else:
                    altrimenti = 1
            else:
                altrimenti = 1
            if altrimenti:
                self.last_idd += 1
                lm.id = self.last_idd
        
        self.lm_t_prec = lm_t
        return gui_lm_track
                
            
        
    

def main():
    START = 0
    STOP = 2
    kitti = Kitti()
    #ekf = Ekf(n=5, m=6, p=0, ml=4)
    #slam = Slam(ekf)
    window = kitti.disegna_gui()  # Disegna l'interfaccia
    ###
    #Ciclo vitale
    ###
    actual_frame = START
    kitti.get_scan(actual_frame)
    #slam.initial_assimilation(scan)
    while True:
        event, values = window.read()
        if event == "Avanti":
            actual_frame += 1
            gui_lm_track = kitti.get_scan(actual_frame)
            kitti.update_gui(actual_frame, window, gui_lm_track)
        if event == "Auto":
            while True and actual_frame < STOP:
                actual_frame += 1
                gui_lm_track = kitti.get_scan(actual_frame)
                kitti.update_gui(actual_frame,window, gui_lm_track)
                time.sleep(0.001)
                event, values = window.read(timeout=1)
        if event == "Stop":
            break
        if event == "Esci" or event == sg.WIN_CLOSED:
            break
    window.close()

if __name__ == "__main__":
    main()
    print("finito, pace e bene.")   

