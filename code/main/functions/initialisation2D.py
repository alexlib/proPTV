'''

    This script contains general functions for the track initialisation routine for 2D images.
    
'''


import joblib, sys, cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import signal

from functions.triangulation import *
from functions.initialisation import *


def Initialisation2D(imgps,t,ax,ay,params):
    # calculate 2D tracks per camera
    allTracks, P_clouds_rest = [], []
    time = np.linspace(t-params.t_init+1,t,params.t_init,dtype=int)
    for c in range(len(params.cams)):
        tracks = np.empty([0,params.t_init,2])
        Ps = imgps[c].copy()
        # for N_init run the initialisation process
        for i in range(params.N_init):
            maxVel = params.maxvel * ((i+1)/params.N_init)
            # initialse tracks forward in time
            tracks_for = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(Linking)(p0, Ps, maxVel, params) for p0 in tqdm(Ps[0],desc='   initialise for: ',position=0,leave=True,delay=0.5))
            tracks_for = np.asarray([track for track in tracks_for if len(track)>0])
            # initialse tracks backwards in time
            tracks_back = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(Linking)(p0,Ps[::-1],maxVel,params) for p0 in tqdm(Ps[::-1][0],desc='   initialise back: ',position=0,leave=True,delay=0.5))
            tracks_back = np.asarray([track for track in tracks_back if len(track)>0])
            # merge tracks
            tracks_for = MergeForwardBackwardLinking(tracks_for,tracks_back)
            if len(tracks_for)>0:
                tracks_for = UniqueFilter(tracks_for,params) 
                Ps = ReducePoints(Ps,tracks_for,params)
                tracks = np.append(tracks,tracks_for,axis=0)
        allTracks.append(tracks)
        P_clouds_rest.append(Ps)
    print('found 2D tracks per camera: ' , str([len(tracks_i) for tracks_i in allTracks]))    
    # load camera images
    cam_imgs = [cv2.imread(params.case_path+"input/raw_images/c{cam}/c{cam}_{time}.tif".format(cam=str(params.cams[c]),time=str(time[0]).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) for cam in params.cams]
    ImgPoints_cams = [tracks[:,0,:] for tracks in allTracks]
    # triangulate 3D tracks
    print('')
    allTracks3D, rest = [], [[[] for ti in range(len(time))] for cam in params.cams]
    for track in tqdm(allTracks[0],desc='Triangulation 3D tracks: ',position=0,leave=True):
        T = GetTriangulationCandidates(track[0],ImgPoints_cams,cam_imgs,np.asarray(ax),np.asarray(ay),params.cams,np.asarray([params.cams]),params)
        if len(T)>0:
            ID = [] 
            for c in range(len(params.cams)):
                ID_c = np.argwhere(np.linalg.norm(T[4+2*c:4+2*(c+1)]-allTracks[c][:,0,:],axis=1)==0)[:,0]
                if len(ID_c)>0:
                    ID.append(ID_c[0])
                else:
                    ID.append(np.nan)
            xy = []
            for j, IDi in enumerate(ID):
                if np.isnan(IDi)==True:
                    xy.append(np.nan*np.ones([len(time),2]))    
                else:
                    for ti in range(len(time)):
                        rest[j][ti].append(allTracks[j][IDi,ti]) 
                    xy.append(allTracks[j][IDi])  
            xy = np.transpose(np.asarray(xy), (1, 0, 2)) 
            track3D = np.asarray([NewtonSoloff_Triangulation(setP, ax, ay, params)[0] for setP in xy])
            allTracks3D.append([list(time),list(track3D),list(Init_Velocity3D(track3D)),list(Init_Acceleration3D(track3D)),[xy[k] for k in range(params.t_init)]])
    # process P_clouds and save out current Triag Points
    rest = [np.asarray(resti) for resti in rest]
    for c in range(len(params.cams)):
        imgps[c] = ReducePoints2(imgps[c],rest[c],params)    
        print('   remaining particles on cam '+str(params.cams[c])+': ', str([len(P) for P in imgps[c]]))
    return allTracks3D, imgps

def Linking(p0,Ps,maxVel,params):
    # first link
    tracks = [np.array([p0,p1]) for p1 in FindNNPoints(p0, Ps[1], params.NN[0])]
    # second and ongoing links via polynom prediction
    for i,P in enumerate(Ps[2::]):
        tracks = np.concatenate([[np.append(track,p.reshape(1,2),axis=0) for p in FindNNPoints((track+Init_Velocity2D(track))[-1],P,params.NN[i+1])] for track in tracks])
    tracks = [track for track in tracks if np.max(np.linalg.norm(np.diff(track,axis=0),axis=1))<maxVel]
    if len(tracks)>0:
        # estimate best track by cost function based on minimal action principle
        E = np.linalg.norm( np.diff( [np.linalg.norm((np.diff(track,axis=0)/np.max(np.abs(np.diff(track,axis=0)))),axis=1)**2 for track in tracks] ,axis=1) ,axis=1)
        theta = np.array( [np.linalg.norm( np.diff( [Theta(np.diff(track,axis=0)[i],np.diff(track,axis=0)[i+1]) for i in range(len(track)-2)] ) ) for track in tracks] )
        track_final = tracks[np.argmin(E*theta)]
        v_final = np.diff(track_final,axis=0)
        Theta_check = np.asarray([np.degrees(np.arccos(np.dot(v_final[i],v_final[i+1]) / (np.linalg.norm(v_final[i])*np.linalg.norm(v_final[i+1]))))<params.angle for i in range(len(v_final)-1)]).all()
        if Theta_check:
            return track_final
    return []

def FindNNPoints(p,P,N):
    # find N nearest neighbour points in P around p
    return P[np.argpartition(np.linalg.norm(p-P,axis=1), N)[:N:]]

def MergeForwardBackwardLinking(tracks_for,tracks_back):
    # change backward tracks to forward time scheme
    tracks_back = np.asarray([track[::-1,:] for track in tracks_back])
    # merge tracks
    tracks = [track for track in tqdm(tracks_for,desc='   merge initialisation: ', position=0, leave=True, delay=0.5) if len(np.argwhere(np.linalg.norm(tracks_back[:,:,:3]-track[:,:3],axis=(1,2))==0)[:,0]) == 1]
    return np.asarray(tracks)

def UniqueFilter(tracks,params):
    # only unique initalised tracks are passed
    IDs = []
    for t in tqdm(range(params.t_init),desc='   uniqueness filter: ', position=0, leave=True, delay=0.5):
        unique, counts = np.unique(tracks[:,t,0], return_counts=True)
        ID = np.argwhere(counts>1)[:,0]
        for i in ID:
            IDs.append(np.argwhere(tracks[:,t,0]==unique[i])[:,0])
    if len(IDs) != 0:
        IDs = np.unique(np.concatenate(IDs))
        tracks = [track for i,track in enumerate(tracks) if i not in IDs]
        return np.asarray(tracks)
    return np.asarray(tracks)

def ReducePoints(P_clouds,tracks,params):
    # reduce the Point clouds by the points occupied in tracks
    xy_used, P_clouds_new = [], []
    for i in tqdm(range(params.t_init),desc='   delete used 2D points: ', position=0, leave=True, delay=0.5):
        deleteList = [np.argwhere( np.linalg.norm(P_clouds[i]-track[i,:2],axis=1)==0)[0][0] for track in tracks]
        xy_used.append( P_clouds[i][deleteList,4::] )
        P_clouds_new.append( np.delete(np.asarray(P_clouds[i]),np.asarray(deleteList),axis=0) if len(deleteList)!=0 else P_clouds[i] )
    return P_clouds_new
def ReducePoints2(P_clouds,tracks,params):
    # reduce the Point clouds by the points occupied in tracks
    xy_used, P_clouds_new = [], []
    for i in tqdm(range(params.t_init),desc='   delete used 2D points: ', position=0, leave=True, delay=0.5):
        deleteList = [np.argwhere( np.linalg.norm(P_clouds[i]-track,axis=1)==0)[0][0] for track in tracks[i]]
        xy_used.append( P_clouds[i][deleteList,4::] )
        P_clouds_new.append( np.delete(np.asarray(P_clouds[i]),np.asarray(deleteList),axis=0) if len(deleteList)!=0 else P_clouds[i] )
    return P_clouds_new

def Theta(v1,v2):
    # estimate the angle between 2 vectors
    v1_norm = np.linalg.norm(v1) if np.linalg.norm(v1)>0 else 1E-8
    v2_norm = np.linalg.norm(v2) if np.linalg.norm(v2)>0 else 1E-8
    return np.degrees(np.arccos(np.dot(v1,v2) / (v1_norm*v2_norm)))

def Init_Position2D(track):
    # estimate the position of a track of arbitory length
    savgol_mode = 'interp' #'nearest'
    pos = np.zeros_like(track)
    pos[:,0] = signal.savgol_filter(track[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=0,mode=savgol_mode)
    pos[:,1] = signal.savgol_filter(track[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=0,mode=savgol_mode)
    return pos

def Init_Velocity2D(track):
    # estimate the velocity of a track of arbitory length
    savgol_mode = 'interp' #'nearest'
    vel = np.zeros_like(track)
    vel[:,0] = signal.savgol_filter(track[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
    vel[:,1] = signal.savgol_filter(track[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=1,mode=savgol_mode)
    return vel

def Init_Acceleration2D(track):
    # estimate the velocity of a track of arbitory length
    savgol_mode = 'interp' #'nearest'
    acc = np.zeros_like(track)
    acc[:,0] = signal.savgol_filter(track[:,0],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
    acc[:,1] = signal.savgol_filter(track[:,1],window_length=min(len(track),5),polyorder=min([len(track)-1,3]),deriv=2,mode=savgol_mode)
    return acc