'''

    Refines calibration based on particle images.
    
'''

import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import least_squares

shutil.copy('../../main/functions/soloff.py', 'functions/')
shutil.copy('../../main/functions/triangulation.py', 'functions/')
from functions.triangulation import *
from functions.soloff import *

os.chdir('../../../data/')


# %%

class Parameter():    
    '''
    case_path, Zeros = 'RBC300/', 7
    cams = [0,1,2,3]
    depthaxis = [1,1,1,1]
    
    subVs_x, subVs_y, subVs_z = [2,3], [2,3], [1,3]
    
    Vmin = [0,0,0]
    Vmax = [300,300,300]
    t0, t1, dt = 15000, 15000, 1   
    
    startCamForPermute = []
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling, Imin = 1, 4, 1.5, 1.5, 0.7, 0, [0,0,0,0]
    overright = True
    
    '''
    case_path, Zeros = 'LPTC_040/', 5
    cams = [0,1,2,3]
    depthaxis = [2,2,2,2]
    
    subVs_x, subVs_y, subVs_z = [2], [2], [2]
    
    Vmin = [-600,-2150,-2500]
    Vmax = [2500,250,500]
    t0, t1, dt = 0, 0, 1   
    
    startCamForPermute = [2]
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling, Imin = 1, 3, 3.0, 1.5, 1.4, 0, [5000,5000,5000,5000]
    overright = True
    #'''
    
# %%

def ReTriangulation(xy,ax,ay,params):
    camPs = [ np.asarray([xy[0][i,:2],xy[1][i,:2],xy[2][i,:2],xy[3][i,:2]]) for i in range(len(xy[0]))]
    return np.asarray([NewtonSoloff_Triangulation(setP, ax, ay, params)[0] for setP in camPs])

def Errors(xyXYZ,P,ax,ay,params):
    XYZ = xyXYZ[0][:,2:]
    dxy = [np.vstack([F(XYZ,ax[i])-xyXYZ[i][:,0],F(XYZ,ay[i])-xyXYZ[i][:,1]]).T for i in range(len(params.cams))]
    dXYZ = XYZ-P
    [print('2D calibration error cam '+str(params.cams[i])+': ', np.round(np.mean(np.linalg.norm(dxy[i],axis=1)),4), ' +- ', np.round(np.std(np.linalg.norm(dxy[i],axis=1)),4) ) for i in range(len(params.cams))]
    print('3D calibration error: ', np.round(np.mean(np.linalg.norm(dXYZ,axis=1)),4), ' +- ', np.round(np.std(np.linalg.norm(dXYZ,axis=1)),4))
    print('')
    return 0

def main(): 
    # load parameter
    params = Parameter()
    params.markerList = params.case_path+"input/calibration_images/markers_c{cam}.txt"
    params.calibration_path = params.case_path+"input/calibration/c{cam}/soloff_c{cam}{xy}.txt"
    params.particle_path = params.case_path+'input/particle_lists/c{cam}/c{cam}_{time}.txt'
    params.triangulation_path = params.case_path+'output/VSC'  
    if params.overright == False:
        params.calibration_output =  params.case_path+'output/VSC/soloff_c{cam}{xy}.txt'
    else:
        params.calibration_output =  params.calibration_path
    
    # load marker
    xyXYZ = [np.loadtxt(params.markerList.format(cam=cam),skiprows=1) for cam in params.cams]
    
    # load intial calibration
    ax = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="x"),delimiter=',') for c in params.cams])
    ay = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="y"),delimiter=',') for c in params.cams])
    
    # estimate initial calibration error
    P = ReTriangulation(xyXYZ,ax,ay,params)
    Errors(xyXYZ,P,ax,ay,params)
    sys.exit()
    
    # particle triangulation
    P_clouds = np.empty([0,4+2*len(params.cams)])
    for t in np.linspace(params.t0,params.t1,params.t1-params.t0+1,dtype=int)[::params.dt]:
        print(' t = ' + str(t) + ':')
        ImgPoints = [np.loadtxt(params.particle_path.format(cam=cam,time=str(t).zfill(params.Zeros)),skiprows=1) for cam in params.cams]
        print('   found particles per cam: ' , str([len(imgP) for imgP in ImgPoints]))
        TriagPoints , ImgPoints = Triangulate3DPoints(ImgPoints,t,ax,ay,params)
        #TriagPoints = np.loadtxt(params.triangulation_path+'/Points_{time}.txt'.format(time=str(t).zfill(params.Zeros)))
        P_clouds = np.append(P_clouds,TriagPoints,axis=0)
        print('   triangulated ' + str(len(TriagPoints)) + ' particles')
        print('   triangulation error: ' + str(np.mean(P_clouds[:,3])))
    print(' finally triangulated ' + str(len(P_clouds)) + ' particles\n')
    
    # set new path to calibration parameters
    params.calibration_path = params.calibration_output
    
    # perform interative refinement
    for run in range(len(params.subVs_x)):
        print('RUN ' + str(run+1),'\n')
        # generate subvolumes
        V_x = np.linspace(params.Vmin[0],params.Vmax[0],params.subVs_x[run]+1)
        V_y = np.linspace(params.Vmin[1],params.Vmax[1],params.subVs_y[run]+1)
        V_z = np.linspace(params.Vmin[2],params.Vmax[2],params.subVs_z[run]+1)
        # subvolume correction and recalibration
        xyXYZ_cor = []
        for c in params.cams:
            xyXYZ_c = np.empty([0,5])
            for xi in np.arange(params.subVs_x[run]):
                for yj in np.arange(params.subVs_y[run]):
                    for zk in np.arange(params.subVs_z[run]):
                        # collect each 3D point inside the subvolume
                        IDs_XYZ = np.argwhere( (V_x[0+xi] <= xyXYZ[c][:,2]) & (xyXYZ[c][:,2] <= V_x[1+xi]) & 
                                               (V_y[0+yj] <= xyXYZ[c][:,3]) & (xyXYZ[c][:,3] <= V_y[1+yj]) & 
                                               (V_z[0+zk] <= xyXYZ[c][:,4]) & (xyXYZ[c][:,4] <= V_z[1+zk]) )[:,0]
                        XYZ_sub = xyXYZ[c][IDs_XYZ,2:]
                        IDs_P = np.argwhere( (V_x[0+xi] <= P_clouds[:,0]) & (P_clouds[:,0] <= V_x[1+xi]) & 
                                             (V_y[0+yj] <= P_clouds[:,1]) & (P_clouds[:,1] <= V_y[1+yj]) & 
                                             (V_z[0+zk] <= P_clouds[:,2]) & (P_clouds[:,2] <= V_z[1+zk]) )[:,0]
                        P_sub = P_clouds[IDs_P,:3]
                        p_sub = np.vstack([F(P_sub,ax[c]),F(P_sub,ay[c])]).T
                        xy_sub = P_clouds[IDs_P,4+(2*c):4+(2*c+2)]
                        # calculate disparity inside the subvolume
                        disparity = p_sub-xy_sub
                        d = np.nanmean( disparity, axis=0 )
                        # estimate accumulator function
                        #xi, yi = np.meshgrid(np.linspace(-1,1,100),np.linspace(-1,1,100))
                        #xi, yi = xi.ravel(), yi.ravel()
                        #A = np.sum([np.maximum( np.zeros_like(xi), 1-((xi-disparity[i,0])**2 + (yi-disparity[i,1])**2)/np.sqrt(0.001) ) for i in range(len(disparity))],axis=0)
                        #d = np.array([xi[np.argmax(A)], yi[np.argmax(A)]])
                        
                        # plot 2D subvolume estimation
                        #print(d,xmax,ymax)
                        #plt.figure()    
                        #plt.plot((p_sub-xy_sub)[:,0],(p_sub-xy_sub)[:,1],'o',c='red')
                        #plt.plot(d[0],d[1],'o',c='green')
                        #plt.contourf(xi.reshape(100,100),yi.reshape(100,100),A.reshape(100,100),cmap='seismic',levels=np.linspace(0,np.max(A),501), norm=TwoSlopeNorm(np.mean(A),vmin=0,vmax=np.max(A)))
                        #plt.plot(xmax,ymax,'o',c='orange')
                        #plt.show()
                        # plot 3D subvolume estimation
                        #axis = plt.figure().add_subplot(projection='3d')
                        #axis.scatter(P_clouds[IDs_P,0],P_clouds[IDs_P,1],P_clouds[IDs_P,2],c='black')
                        #axis.scatter(XYZ_sub[:,0],XYZ_sub[:,1],XYZ_sub[:,2],c='red')
                        #axis.set_xlim(params.Vmin[0],params.Vmax[0]), axis.set_ylim(params.Vmin[1],params.Vmax[1]), axis.set_zlim(params.Vmin[2],params.Vmax[2])
                        #plt.show()
                        
                        # correct reprojection and add xyXYZ of the subvolume
                        #xy_cor = np.vstack([ (F(P_sub,ax[c])-d[0]) , (F(P_sub,ay[c])-d[1]) ]).T
                        #xyXYZ_c = np.append(xyXYZ_c,np.append(xy_cor,P_sub,axis=1),axis=0)
                        xy_cor = np.vstack([ (F(XYZ_sub,ax[c])-d[0]) , (F(XYZ_sub,ay[c])-d[1]) ]).T
                        xyXYZ_c = np.append(xyXYZ_c,np.append(xy_cor,XYZ_sub,axis=1),axis=0)
            # add corrected xyXYZ for each camera 
            xyXYZ_cor.append(xyXYZ_c)        
            # least square estimation of the Soloff parameter
            xy, XYZ = xyXYZ_c[:,:2], xyXYZ_c[:,2:]
            ax[c] = least_squares(lambda a: F(XYZ,a)-xy[:,0], ax[c], method='trf').x
            ay[c] = least_squares(lambda a: F(XYZ,a)-xy[:,1], ay[c], method='trf').x
            # save out new calibration
            np.savetxt(params.calibration_output.format(cam=c,xy='x'),ax[c])
            np.savetxt(params.calibration_output.format(cam=c,xy='y'),ay[c])
             
        # estimate refined calibration error    
        P = ReTriangulation(xyXYZ_cor,ax,ay,params)
        Errors(xyXYZ_cor,P,ax,ay,params)
        
        # triangulation of particles
        P_clouds = np.empty([0,12])
        for t in np.linspace(params.t0,params.t1,params.t1-params.t0+1,dtype=int)[::params.dt]:
            print(' t = ' + str(t) + ':')
            ImgPoints = [np.loadtxt(params.particle_path.format(cam=cam,time=str(t).zfill(params.Zeros)),skiprows=1) for cam in params.cams]
            print('   found particles per cam: ' , str([len(imgP) for imgP in ImgPoints]))
            TriagPoints , ImgPoints = Triangulate3DPoints(ImgPoints,t,ax,ay,params)
            P_clouds = np.append(P_clouds,TriagPoints,axis=0)
            print('   triangulated ' + str(len(TriagPoints)) + ' particles')
            print('   triangulation error: ' + str(np.mean(P_clouds[:,3])))
        print(' finally triangulated ' + str(len(P_clouds)) + ' particles\n')
if __name__ == "__main__":
    main()