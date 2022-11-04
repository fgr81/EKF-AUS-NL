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
        x=2.*random.random()-1
        y=2.*random.random()-1
        r=x*x+y*y;
        if r<1. and r!=0.:
            fac = math.sqrt(-2.0*math.log(r)/r)
            # print(f"gaussiano: {x*fac}")
            return x*fac
            
def nonlinH(state):
    
    p = state.shape[0]

    out = np.zeros((p,1), dtype=float, order='F')

    for i in range(p):
        out[i] = state[i]

    return out # misuro tutto con un certo errore


def write_file(nf, xf):
    f = open(nf, "w")

    nn = xf.shape[0]

    f.write(f"{nn} \n")

    for i in range(nn):
        f.write(f"{xf[i]}\n")
    f.close()

def read_file(nf):
    with open (nf, "r") as myfile:
        content = myfile.readlines()
    myfile.close()

    N0 = int(content[0])
    xf = np.zeros((N0,1), dtype=float, order='F')

    for i in range(N0):
        xf[i] = float(content[i+1])

    return xf
    

def evolve(XX, time):
    
    comando = "./external/my_l96/MyL96 statotemporaneo.dat " + str(time)  

    for k in range(0,XX.shape[1]):      
        write_file("statotemporaneo.dat", XX[:,k])
        os.system(comando)
        colonna = read_file("statotemporaneo.dat")
        for i in range(0,colonna.shape[0]):
            XX[i,k] = colonna[i]
    return XX


print("Script di test ekfaus.py con il modello Lorenz96")

Assimilationcycles = 1000


npert = 16

truth = read_file('truthL96.dat')

N0 = int(truth.shape[0])

print("N0 = ",N0)


#################################################################
#
# Initializing the EKF_AUS_NL class with N0 degrees of freedom,
# 15 linear Lyapunov vectors, N0 number of measurements, 
# no nonlinear interaction
#
################################################################

ekf_aus = ekf.EkfAus(N0, npert, N0, 4) 
ekf_aus.AddInflation(1.e-12)
ekf_aus.LFactor(0.005)


# ekf_aus.Lfactor(0.01)

nc = ekf_aus.TotNumberPert()

print("# Total number of perturbation = ",nc)

sigmad = 0.3 # errore di misura

print("# Measurement error = ",sigmad)
print("# number of measurements p = ",ekf_aus.P())

R = np.zeros((ekf_aus.P(),1))
for i in range(ekf_aus.P()):
    R[i,0] = sigmad * sigmad

Xa =  sigmad * np.random.rand(N0, nc)

#print(Xa)

gmunu = np.ones((N0,1), dtype=float, order='F')

analysis = np.zeros((N0,1))

for i in range(N0):
    g = gaussiano()
    analysis[i, 0] = truth[i,0] + sigmad * g 

#print(truth)
#print(analysis)

    
traj = open("traiettorieL96.dat", "w")
alog = open("perfectL96.log", "w")


for kk in range(Assimilationcycles):
    #print(f"DENTRO IL CICLO, KK={kk}")

    #ekf_aus.SetModelErrorVariable( N0 - 2, N0 - 1, ModelError, Xa)

    perturbazioni = ekf_aus.PrepareForEvolution( analysis, Xa, gmunu)
    errore = analysis - truth

    #errore /= np.max(np.abs(errore),axis=0)  # normalizzazione
    _log = f"Timestep= {kk} Analysis Error= {np.linalg.norm(errore) / math.sqrt(N0)} \n"
    alog.write(_log)

    perturbazioni = evolve(perturbazioni, 0.2)
    analysis = evolve(analysis, 0.2)
    truth = evolve(truth, 0.2)

    errore = analysis - truth
#    errore /= np.max(np.abs(errore),axis=0)
    _log = f"Timestep = {kk} Forecast Error = {np.linalg.norm(errore) / math.sqrt(N0)} \n"
    alog.write(_log)

    measure = nonlinH(truth)

    for i in range(ekf_aus.P()):
        g = gaussiano()
        measure[i, 0] += sigmad * g 

    Xa = ekf_aus.PrepareForAnalysis(analysis, perturbazioni, gmunu)

#    print(analysis)

    ekf_aus.Assimilate(measure, nonlinH, R, analysis, Xa)

#    print(analysis)

    traj.write(f"{truth[0, 0]} {truth[1, 0]} {truth[2, 0]} {analysis[0, 0]} {analysis[1, 0]} {analysis[2, 0]}\n")
    print(f"Timestep = {kk} Forecast Error = {np.linalg.norm(errore) / math.sqrt(N0)}")
    
traj.close()
alog.close()

