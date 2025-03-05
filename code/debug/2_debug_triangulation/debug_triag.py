'''

    Debug triangulation.
    
'''


import os, cv2, shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

shutil.copy('../../main/functions/triangulation.py', 'functions/')
shutil.copy('../../main/functions/soloff.py', 'functions/')

from functions.triangulation import *
from functions.soloff import *

os.chdir('../../../data')


# %%

class TriagParameter():
    case_name, runname, Zeros = "LPTC_040", "run1", 5
    cams, t = [0,1,2,3], 0
    depthaxis = [2,2,2,2]
    
    d, mode = 0.0075, 'run'
    n = 0
    
    Vmin , Vmax = [-600,-600,-600], [600,600,600]    
    startCamForPermute = []
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling, Imin = 1, 4, 2.0, 1.0, 0.4, 0, [1000,1000,1000,1000]
    
# %%


def main(): 
    # load parameter
    params = TriagParameter()
    params.case_path = params.case_name+"/" 
    params.calibration_path = params.case_path + "input/calibration/c{cam}/soloff_c{cam}{xy}.txt" 
    par = params.case_path + "input/particle_lists/c{cam}/c{cam}_{timeString}.txt" 
    params.img_path = params.case_path + "input/raw_images/c{c}/c{c}_{time}.tif"
    params.triangulation_path = "../code/debug/2_debug_triangulation/output"  
    
    # load calibration
    ax = [np.loadtxt(params.case_name+'/input/calibration/c{cam}/soloff_c{cam}{xy}.txt'.format(cam=cam,xy="x"),delimiter=',') for cam in params.cams]
    ay = [np.loadtxt(params.case_name+'/input/calibration/c{cam}/soloff_c{cam}{xy}.txt'.format(cam=cam,xy="y"),delimiter=',') for cam in params.cams]
    # load img points
    imgPs = np.asarray([np.loadtxt(params.case_name+'/input/particle_lists/c{cam}/c{cam}_{time}.txt'.format(cam=c,time=str(params.t).zfill(params.Zeros)),skiprows=1) for c in params.cams],dtype=object)
    # load images
    imgs_cams = [cv2.imread(params.img_path.format(c=cam,time=str(params.t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) for cam in params.cams]
    # triangulation
    print('Triangulation:')
    if params.mode == 'load':
        Triag = np.loadtxt(params.case_name+"/output/"+params.runname+"/triangulation/Points_{time}.txt".format(time=str(params.t).zfill(params.Zeros)),skiprows=1)
    else:
        Triag , imgPs_new = Triangulate3DPoints(imgPs,params.t,ax,ay,params)
    print(' triangulated ' + str(len(Triag)) + ' points from ' + str([len(ImgP) for ImgP in imgPs]) + ' image points' )
    print(' triangulation error: ' + str(np.mean(Triag[:,3])))
    
    # debug triangulation
    # 3D plot
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    axis.scatter(Triag[:,0],Triag[:,1],Triag[:,2],color='red',s=0.1)
    axis.set_xlim(params.Vmin[0],params.Vmax[0]), axis.set_ylim(params.Vmin[1],params.Vmax[1]), axis.set_zlim(params.Vmin[2],params.Vmax[2])
    plt.show()
    # 2D plot
    i = 0
    n = [0,1,0,1]
    m = [0,0,1,1]
    fig, axis = plt.subplots(2,2,sharex=True,sharey=True,figsize=(12,12))
    for c in params.cams:
        xy_sub = Triag[:,4+(2*c):4+(2*c+2)]
        x , y = F(Triag[:,:3], ax[c]) , F(Triag[:,:3], ay[c])
        axis[n[i]][m[i]].imshow(imgs_cams[c],cmap='gray',vmax=10000)
        axis[n[i]][m[i]].plot(imgPs[c][:,0],imgPs[c][:,1],'o',c='orange')
        axis[n[i]][m[i]].plot(x,y,'.',c='red')
        axis[n[i]][m[i]].plot(xy_sub[:,0],xy_sub[:,1],'.',c='green')
        i+=1
    plt.tight_layout(), plt.show()
    
    # debug triangulation with ground truth
    if os.path.isfile(params.case_name+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros))):
        Ps = np.loadtxt(params.case_name+'/analysis/origin/origin_{time}.txt'.format(time=str(params.t).zfill(params.Zeros)),skiprows=1)[:,1::]
        counts, dels, dels_Triag, wrongID = 0, [], [] ,[]
        for i , p in enumerate(tqdm(Triag,desc='Debug Triangulation: ',position=0,leave=True,delay=0.5)):
            positions, imgpoints = p[:3:], p[4::]
            dP = np.linalg.norm(positions-Ps[:,:3:],axis=1)
            ID = np.argmin(dP)
            if dP[ID]<params.d:
                if not ID in dels:
                    counts += 1
                dels.append(ID)
                dels_Triag.append(i)
            else:
                wrongID.append(dP[ID])
        print(len(dels))
        print(len(np.unique(dels)))
        print(len(dels_Triag))
        print(len(np.unique(dels_Triag)))
        Ps_del = np.delete(Ps,dels,axis=0)
        Triag_del = np.delete(Triag,dels_Triag,axis=0)
        T1 = Triag[np.asarray(dels_Triag)]
        T2 = Ps[np.asarray(dels)]
        print('\n found ' + str(counts) + ' / ' + str(len(Ps)) + ' ( '+ str(round(counts/len(Ps)*100,2)) +' % ) true points')
        print(' found ' + str(len(Triag)-counts) + ' / ' + str(len(Triag)) + ' ( '+ str(round((len(Triag)-counts)/len(Triag)*100,2)) +' % ) wrong points')
        # plot wrong ID hist
        plt.figure()
        plt.hist(wrongID,bins=50)
        plt.show()
        # 3D plot
        fig = plt.figure()
        axis = fig.add_subplot(111, projection='3d')
        axis.scatter(T1[:,0],T1[:,1],T1[:,2],color='red',s=0.8)
        axis.scatter(T2[:,0],T2[:,1],T2[:,2],color='green',s=0.8)
        plt.show()
        # plot points
        fig, axis = plt.subplots(1,2,figsize=(16,8),subplot_kw=dict(projection='3d'))
        axis[0].scatter(Triag[:,0],Triag[:,1],Triag[:,2],color='red',s=0.8)
        axis[0].scatter(Ps[:,0],Ps[:,1],Ps[:,2],color='green',s=0.8)
        axis[1].scatter(Triag_del[:,0],Triag_del[:,1],Triag_del[:,2],color='red',s=0.8)
        axis[1].scatter(Ps_del[:,0],Ps_del[:,1],Ps_del[:,2],color='green',s=0.8)
        plt.show()
if __name__ == "__main__":
    main()