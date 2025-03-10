import os, sys, h5py
import numpy as np

from tqdm import tqdm

os.chdir('../../main')
from functions.setup import *
from functions.initialisation import *

os.chdir('../../data')


# %%

class parameter(): 
    case_name, runname, suffix, ending = 'RBC300', 'run1_backtracking', '_backtracking', ''
    t_start, t_end, dt = 15000, 15100, 1
    loadBroken = False
    f = 5 # Hz
    
# %%


def main(): 
    params = parameter()
    params.track_path = params.case_name+'/output/'+params.runname+'/tracks/'
    params.lagrange_path = params.case_name+'/output/'+params.runname+'/PINN/Lagrange_{time}.txt'
    
    # load tracks
    allTracks = LoadTracks(params.track_path,params.suffix)
    if params.loadBroken == True:
        for t in np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int):
            if os.path.isfile(params.case_name+'/output/'+params.runname+'/tracks/tracks_broken{time}.hdf5'.format(time=t)):
                allTracks += LoadTracks(params.case_name+'/output/'+params.runname+'/tracks/','_broken{time}'.format(time=t))
    print(' loaded ' + str(len(allTracks)) + ' tracks')
    
    allTracks_new = []
    for track in tqdm(allTracks,leave=True,position=0):
        if len(track)>6:
            track_new = np.zeros([len(track),9])
            track_new[:,0] = (track[:,0]-params.t_start)/params.f
            track_new[:,1:4] = Init_Position3D(track[:,1:4])
            track_new[:,4:7] = Init_Velocity3D(track[:,1:4])*params.f
            allTracks_new.append(track_new)
    print(' finalized ' + str(len(allTracks_new)) + ' tracks')
                
    os.makedirs( params.case_name+'/output/'+params.runname+'/PINN' , exist_ok=True)
    for i,t in enumerate(tqdm((np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)-params.t_start)/params.f,leave=True,position=0)):
        Lagrange = []
        for track in allTracks_new:
            ID = np.argwhere(track[:,0]==t)[:,0]
            if len(ID)>0:
                Lagrange.append(track[ID[0]])
        if len(Lagrange)==0:
            print('Fail - no points in this time step')
            sys.exit()
        np.savetxt(params.lagrange_path.format(time=str(i)),np.asarray(Lagrange),header='t[s] , x[mm] , y[mm] , z [mm], u[mm/s] , v[mm/s] , w[mm/s] , T=0 , p=0 ')
if __name__ == "__main__":
    main()