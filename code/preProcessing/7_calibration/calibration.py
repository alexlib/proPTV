'''
    This script does the calibration based on an marker list per camera.
'''


import os
import numpy as np

from scipy.optimize import least_squares

os.chdir('../../../code/main/functions')

from soloff import *

os.chdir("../../../data/")


# %%

class Calibration_parameter:
    case_name = 'LPTC_040'
    cams = [0,1,2,3]
    N = 20

# %%
  
  
def main():
    params = Calibration_parameter()
    params.markerList_input = params.case_name+"/input/calibration_images/markers_c{cam}.txt"
    params.calibration_output = params.case_name+"/input/calibration/c{cam}/soloff_c{cam}{xy}.txt"
    
    # load marker lists for each camera
    for cam in params.cams:
        xyXYZ = np.loadtxt(params.markerList_input.format(cam=cam),skiprows=1)
        xy, XYZ = xyXYZ[:,:2:], xyXYZ[:,2::]
        
        # least square estimation of the Soloff parameter
        sx = least_squares(lambda a: F(XYZ,a)-xy[:,0], np.zeros(params.N), method='trf').x
        sy = least_squares(lambda a: F(XYZ,a)-xy[:,1], np.zeros(params.N), method='trf').x
        
        np.savetxt(params.calibration_output.format(cam=cam,xy="x"),sx)
        np.savetxt(params.calibration_output.format(cam=cam,xy="y"),sy)
if __name__ == "__main__":
    main()