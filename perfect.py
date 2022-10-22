import os
import math
import random
import numpy as np
import ekfaus as ekf



def gaussiano():
    fac = 0.
    x = 0.
    y = 0.
    r = 0.
    while(fac==0):
        # x=2*ranf_arr_next()-1
        # y=2*ranf_arr_next()-1
        x=2*random.random()-1
        y=2*random.random()-1
        r=x*x+y*y;
        if(r<1. and r!=0.):
            fac = math.sqrt(-2.0*math.log(r)/r)
            # print(f"gaussiano: {x*fac}")
            return x*fac
            
'''
@brief: kkk 
@state Lo state variables are sorted in this way
x, y, phi, x1, y1, x2, y2, x3, y3, ......, V, G
'''
def nonlinH(state):
    nm = len(state) - 5
    misura = np.zeros((nm,1), dtype=float, order='F')
    for i in range(3, nm + 3, 2):
        dx = state[i,0] - state[0,0]
        dy = state[i + 1, 0] - state[1,0]
        d = math.sqrt(dx * dx + dy * dy)
        misura[i - 3, 0] = d
        misura[i - 2, 0] = math.atan2(dx, dy) - state[2,0]
    return misura

def InitializeEKF_SLAM( N0, nc, R, firstMeasure):
    state = np.zeros((N0,1), dtype=float, order='F')
    Xa = np.zeros((N0, nc), dtype=float, order='F')
    r = theta = a = b = c = d = 0.
    n = 0
    #state[ 0, 0] = state[ 1, 0] = state[ 2, 0] = 0.
    for i in range(0, np.shape(firstMeasure)[0], 2):
        r = firstMeasure[ i, 0]
        theta = firstMeasure[ i + 1, 0]
        state[ i + 3, 0] = r * math.cos( theta + state[ 2, 0])
        state[ i + 4, 0] = r * math.sin( theta + state[ 2, 0])
        for j in range( 0, np.shape(Xa)[1], 1):
            a = gaussiano()
            b = gaussiano()
            c = gaussiano()
            d = gaussiano()
            Xa[ i+3, j] = a * math.sqrt( R[ i, 0 ]) * math.cos( theta + state[ 2, 0]) + b * math.sqrt( R[ i+1, 0] ) * math.sin( theta + state[2,0])
            Xa[ i+4, j] = c * math.sqrt( R[ i, 0 ]) * math.sin( theta + state[ 2, 0]) + d * math.sqrt( R[ i+1, 0] ) * math.cos( theta + state[2,0])
    return state, Xa


def write_file(nf, xf):
    f = open(nf, "w")
    f.write(f"{xf[0]}\n{xf[1]}\n{xf[2]}\n{xf[-2]}\n{xf[-1]}\n")
    n = int((xf.shape[0] - 5) / 2)
    f.write(f"{n}\n")
    for i in range(n):
        f.write(f"{xf[3 + 2 * 1]} {xf[4 + 2 * i]}\n")
    f.close()

def read_file(nf):
    with open (nf, "r") as myfile:
        k = myfile.readlines()
    x = float(k[0])
    y = float(k[1])
    phi = float(k[2])
    V = float(k[3])
    G = float(k[4])
    N = int(k[5])
    X = np.zeros(( 2 * N + 5, 1), dtype=float, order='F')
    X[0,0] = x
    X[1,0] = y
    X[2,0] = phi
    for i in range(N):
        kk = k[ 6 + i].split()
        X[3 + 2 * i, 0] = kk[0]
        X[4 + 2 * i, 0] = kk[1]
    myfile.close()
    return X
    

def evolve(XX, time):
    nsteps = int(time / 0.025)
    comando = "./MySlam statotemporaneo.dat " + str(nsteps)  
    for k in range(0,XX.shape[1]):      
        write_file("statotemporaneo.dat", XX[:,k])
        os.system(comando)
        colonna = read_file("statotemporaneo.dat")
        for i in range(0,colonna.shape[0]):
            XX[i,k] = colonna[i]
    return XX

          
'''
#####
 il file truth è nella forma:
 X iniz
 Y iniz
 theta iniz
 V iniz
 vel rad iniz
 num lm
 [lm_x lm_y]
#####
'''
Assimilationcycles = 5000
with open ('truth', "r") as myfile:
    ftruth = myfile.readlines()
myfile.close()
N0 = ((len(ftruth) - 6 ) * 2 ) + 5
truth = np.zeros((N0,1), dtype=float, order='F')
startpert = np.zeros((N0,1), dtype=float, order='F')
VV = float(ftruth[3])
GG = float(ftruth[4])
truth[0] = float(ftruth[0])
truth[1] = float(ftruth[1])
truth[2] = float(ftruth[2])
ii = 3
for i in range(6, len(ftruth)):
    k = ftruth[i].split()
    truth[ii] = k[0]
    truth[ii + 1] = k[1]
    ii += 2
truth[-2] = VV
truth[-1] = GG
#####
# Initializing the EKF_AUS_NL class with N0 degrees of freedom,
# 6 linear Lyapunov vectors, N0-2 number of measurements, 
# no nonlinear interaction
#####
ekf_aus = ekf.EkfAus(N0, 6, N0 - 5, 0) 
ekf_aus.AddInflation(1.e-12)
# ekf_aus.Lfactor(0.01)
nc = ekf_aus.TotNumberPert()
sigmad = 0.20
sigmaA = 3 * math.pi / 360. # 1 grado sessagesimale
R = np.zeros((ekf_aus.P(),1))
for i in range(ekf_aus.P()):
    if(i%2): #  odd: angle 
        R[i,0] = sigmaA * sigmaA
    else:   # even: distance 
        R[i,0] = sigmad * sigmad
gmunu = np.ones((N0,1), dtype=float, order='F')
ModelError = np.ones((2,1), dtype=float, order='F') 
ModelError[0,0] = 0.05 # (m/s) error in the velocity  
ModelError[1,0] = 3.*math.pi/180. #  3 degrees error for the steering angle 
measure = nonlinH(truth);
for i in range(ekf_aus.P()):
    g = gaussiano()
    measure[i, 0] += math.sqrt(R[i, 0]) * g
    
print("Initializing analyis and Xa")
analysis, Xa = InitializeEKF_SLAM( N0, nc, R, measure)
traj = open("traiettorie.dat", "w")
alog = open("perfect.log", "w")

for kk in range(Assimilationcycles):
    print(f"DENTRO IL CICLO, KK={kk}")
    ekf_aus.SetModelErrorVariable( N0 - 2, N0 - 1, ModelError, Xa)
    truth[ N0 - 2, 0] = VV
    truth[ N0 - 1, 0] = GG
    analysis[ N0 - 2, 0] = VV + ModelError[ 0, 0] * gaussiano()
    analysis[ N0 - 1, 0] = GG + ModelError[ 1, 0] * gaussiano()
    perturbazioni = ekf_aus.PrepareForEvolution( analysis, Xa, gmunu)
    errore = analysis - truth
    errore /= np.max(np.abs(errore),axis=0)  # normalizzazione
    _log = f"Timestep= {kk} Analysis Error= {errore / math.sqrt(N0)} \n"
    alog.write(_log)
    perturbazioni = evolve(perturbazioni, 0.05)
    analysis = evolve(analysis, 0.05)
    truth = evolve(truth, 0.05)
    errore = analysis - truth
    errore /= np.max(np.abs(errore),axis=0)
    _log = f"Timestep = {kk} Forecast Error = {errore / math.sqrt(N0)} \n"
    alog.write(_log)
    Xa = ekf_aus.PrepareForAnalysis(analysis, perturbazioni, gmunu)
    measure = nonlinH(truth)
    ekf_aus.Assimilate(measure, nonlinH, R, analysis, Xa)
    traj.write(f"{truth[0, 0]} {truth[1, 0]} {truth[2, 0]} {analysis[0, 0]} {analysis[1, 0]} {analysis[2, 0]}\n")
    
traj.close()
alog.close()

