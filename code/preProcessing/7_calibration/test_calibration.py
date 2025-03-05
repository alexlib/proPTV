'''
    This script runs a calibration test.
'''

import os, cv2
import numpy as np
import matplotlib.pyplot as plt

os.chdir('../../../code/main/functions')

from soloff import *

os.chdir("../../../data/")

# %%

class TestCalibrationParameter:
    case_name = "LPTC_040"
    cams = [0,1,2,3]
    x0 , x1 , Nx = -600 , 600 , 10
    y0 , y1 , Ny = -600 , 600 , 10
    z0 , z1 , Nz = -600 , 600 , 1       

# %%

def main():
    params = TestCalibrationParameter()
    #params.rawimage_input = params.case_name+"/input/calibration_images/c{cam}/c{cam}_1_01.tif"
    params.rawimage_input = params.case_name+"/input/raw_images/c{cam}/c{cam}_00000.tif"
    params.markerList_input = params.case_name+"/input/calibration_images/markers_c{cam}.txt"
    params.calibration_path = params.case_name+"/input/calibration/c{cam}/soloff_c{cam}{xy}.txt"
    
    ax = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="x"),delimiter=',') for c in params.cams])
    ay = np.asarray([np.loadtxt(params.calibration_path.format(cam=c,xy="y"),delimiter=',') for c in params.cams])

    i = 0
    n = [0,1,0,1]
    m = [0,0,1,1]
    
    # 2D plot
    fig, axis = plt.subplots(2,2,sharex=True,sharey=True)
    for c in params.cams:
        X , Y , Z = np.meshgrid(np.linspace(params.x0,params.x1,params.Nx),np.linspace(params.y0,params.y1,params.Ny),np.linspace(params.z0,params.z1,params.Nz))
        XYZ = np.vstack([np.ravel(X),np.ravel(Y),np.ravel(Z)]).T
        #XYZ = np.loadtxt(params.markerList_input.format(cam=c),skiprows=1)[:,2::]        
        x , y = F(XYZ, ax[c]) , F(XYZ, ay[c])
        img = cv2.imread(params.rawimage_input.format(cam=c),cv2.IMREAD_UNCHANGED)
        axis[n[i]][m[i]].imshow(img,cmap='gray')
        axis[n[i]][m[i]].plot(x,y,'.',c='red')
        i+=1
        
        # save synthetic marker points
        #np.savetxt(params.case_name+"/input/calibration_images/synmarkers_c{cam}.txt".format(cam=c),np.append(np.vstack([x,y]).T,XYZ,axis=1),header='x,y,X,Y,Z')
        
    plt.tight_layout(), plt.show()
if __name__ == "__main__":
    main()