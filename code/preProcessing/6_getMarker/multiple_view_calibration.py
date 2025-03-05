import cv2, sys, os
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

os.chdir('../../../data/')

# %%

class Parameter:
    case_name = 'Ilmenau'
    cams, planes = [0,1,2,3], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    
    marker_distance, Z0 = (120, 120), 0 # X[mm], Y[mm], Z0[mm] , shift -360 in x and +360 in y for davis
    marker_size = (8, 7)  # x, y
    image_size = (2560, 2048)  # x[px], y[px]

# %%
    
# by https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
def DLT(P1, P2, point1, point2):
    A = np.array([point1[1]*P1[2,:]-P1[1,:], P1[0,:]-point1[0]*P1[2,:], point2[1]*P2[2,:]-P2[1,:], P2[0,:]-point2[0]*P2[2,:]]).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
    return Vh[3,0:3]/Vh[3,3]

def main():
    # convention marker sort: left to right (x) then top to bottom (y)
    # note that y top denotes to the smallest y because in the image the y axis is inversed+
    # Therefore, in the images the top left point is (0,0,0) in the first frame
    params = Parameter()   
    params.markerList = params.case_name+"/input/calibration_images/c{cam}/marker/c{cam}_{plane}.txt" 
    params.markerList_out = params.case_name+"/input/calibration_images/c{cam}/marker/c{cam}_{plane}_xyXYZ.txt"
    
    # define first plane XYZ as reference with Z = 0
    X, Y, Z = np.meshgrid(np.arange(0,params.marker_size[0]*params.marker_distance[0],params.marker_distance[0]),-np.arange(0,params.marker_size[1]*params.marker_distance[1],params.marker_distance[1]),np.linspace(params.Z0,params.Z0,1))
    XYZ = [np.asarray(np.vstack([X.ravel(),Y.ravel(),Z.ravel()]).T, dtype=np.float32) for t in params.planes]
    
    # calibrate cameras
    xy_c, R_c, ret_c, M_c, d_c, r_c, t_c, pos_c = [], [], [], [], [], [], [], []
    for c in params.cams:
        xy = [np.asarray(np.loadtxt(params.markerList.format(cam=c,plane=str(t)),skiprows=1), dtype=np.float32) for t in params.planes]
        xy_c.append(xy)
        ret, M, d, r, t = cv2.calibrateCamera(XYZ,xy,params.image_size,None,None) 
        ret_c.append(ret), M_c.append(M), d_c.append(d), r_c.append(r), t_c.append(t)
        R = cv2.Rodrigues(r[0])[0] 
        R_c.append(R)
        pos = -np.dot(R.T, t[0]).ravel()
        pos_c.append(pos)
        print('position cam '+str(c)+': ', pos)
    
    # stereo matching of 3D marker positons - use only cam 0 and cam 1
    ret, CM0, dist0, CM1, dist1, R, T, E, F = cv2.stereoCalibrate(XYZ[0:1], xy_c[0][0:1], xy_c[1][0:1], M_c[0], d_c[0], M_c[1], d_c[1], params.image_size)
    #projection matrix for camera 0
    RT0 = np.concatenate([R_c[0], t_c[0][0]], axis = -1)
    P0 = M_c[0] @ RT0 
    #projection matrix for camera 1
    RT1 = np.concatenate([R@R_c[0], (R@t_c[0][0]+T)], axis = -1)
    P1 = M_c[1] @ RT1 
    # P contains the 3D marker positions of each plate at each position
    P = [np.asarray( [DLT(P0, P1, xy0, xy1) for xy0, xy1 in zip(xy_c[0][i],xy_c[1][i])] ,dtype=np.float32) for i in range(len(params.planes))]
    
    # recalibrate cameras
    for i,c in enumerate(params.cams):
        ret, M, d, r, t = cv2.calibrateCamera(P,xy_c[i],params.image_size,M_c[i],d_c[i],flags=cv2.CALIB_USE_INTRINSIC_GUESS)
        ret_c[i], M_c[i], d_c[i], r_c[i], t_c[i] = ret, M, d, r, t
    
    # stereo matching of 3D marker positons - use only cam 0 and cam 1
    ret, CM0, dist0, CM1, dist1, R, T, E, F = cv2.stereoCalibrate(XYZ[0:1], xy_c[0][0:1], xy_c[1][0:1], M_c[0], d_c[0], M_c[1], d_c[1], params.image_size)
    #projection matrix for camera 0
    RT0 = np.concatenate([R_c[0], t_c[0][0]], axis = -1)
    P0 = M_c[0] @ RT0 
    #projection matrix for camera 1
    RT1 = np.concatenate([R@R_c[0], (R@t_c[0][0]+T)], axis = -1)
    P1 = M_c[1] @ RT1 
    # P contains the 3D marker positions of each plate at each position
    P = [np.asarray( [DLT(P0, P1, xy0, xy1) for xy0, xy1 in zip(xy_c[0][i],xy_c[1][i])] ,dtype=np.float32) for i in range(len(params.planes))]
    
    # save out Soloff dataset
    for c in range(len(params.cams)):
        for i in range(len(params.planes)):
            output = np.append(xy_c[c][i],P[i],axis=1)
            np.savetxt(params.markerList_out.format(cam=str(params.cams[c]),plane=str(params.planes[i])), output, header='x,y,X,Y,Z')
    
    #'''
    # plot 3D position
    fig = plt.figure(figsize=(12,12))
    axis = fig.add_subplot(111, projection='3d')
    axis.set_xlabel('Z [mm]'), axis.set_ylabel('X [mm]'), axis.set_zlabel('Y [mm]')
    axis.set_xlim(-3000,5000), axis.set_ylim(-3500,4500), axis.set_zlim(-2150,250) # 2.38 x 7.0 m , Z=0 -> 4.5 # 20cm bis oben und unten
    axis.scatter(pos_c[0][2],pos_c[0][0],pos_c[0][1],label='c0')  
    axis.scatter(pos_c[1][2],pos_c[1][0],pos_c[1][1],label='c1')  
    axis.scatter(pos_c[2][2],pos_c[2][0],pos_c[2][1],label='c2')  
    axis.scatter(pos_c[3][2],pos_c[3][0],pos_c[3][1],label='c3')  
    axis.scatter(XYZ[0][:,2],XYZ[0][:,0],XYZ[0][:,1],c='red',label='first plane')   
    [axis.scatter(P[i][:,2],P[i][:,0],P[i][:,1],c='green') for i in range(len(params.planes))]
    theta = np.linspace(0, 2*np.pi, 100)
    geometry_down = [3500*np.cos(theta)+1000,3500*np.sin(theta)+480,np.zeros_like(theta)-2150]
    geometry_up = [3500*np.cos(theta)+1000,3500*np.sin(theta)+480,np.zeros_like(theta)+250]
    axis.plot(geometry_down[0],geometry_down[1],geometry_down[2],c='black')
    axis.plot(geometry_up[0],geometry_up[1],geometry_up[2],c='black')
    axis.plot(1000,480,-2150,'x',c='black')
    axis.plot(1000,480,250,'x',c='black')
    plt.legend()
    plt.show()
    '''
    # plot 2D position
    p, _ = cv2.projectPoints(P[0], r_0[0], t_0[0], M_0, d_0)
    p = p.reshape(56,2)
    img = cv2.imread("../../Calibration/c{cam}_{time}.tif".format(cam=0,time=str(1).zfill(5)),cv2.IMREAD_UNCHANGED)
    plt.figure()
    plt.imshow(img,cmap='gray')
    plt.plot(p[:,0],p[:,1],'o',c='red')
    plt.plot(xy_0[0][:,0],xy_0[0][:,1],'o',c='blue')
    plt.show()
    '''
if __name__ == "__main__":
    main()
    
    # undistortion process test -> cv2