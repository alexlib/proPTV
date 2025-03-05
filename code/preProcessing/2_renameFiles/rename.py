'''

    Script to rename files.
    
'''

import os
import numpy as np

# %%

class Rename_parameter:
    # case name
    name, folder = "LPTC_040", "raw_images"
    # file characteristics, like name, time strings etc.
    t_start, t_end = 0, 49
    cam_input, cam_output = [0,1,2,3],  [0,1,2,3]
    oldName, newName = "LPT_CASE03_TR_ppp_0_040_I{time}_{cam}.tif", "c{cam}_{time}.tif"
    Zeros_old, Zeros_new = 4, 5

# %%

def main():
    # load parameter
    params = Rename_parameter()
    # rename files
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)
    for i in range(len(params.cam_input)):
        for t in times:
            os.rename("../../../data/{name}/input/{folder}/c{cam}/".format(folder=params.folder,name=params.name,cam=params.cam_output[i]) + params.oldName.format(cam=params.cam_input[i],time=str(t).zfill(params.Zeros_old)), 
                      "../../../data/{name}/input/{folder}/c{cam}/".format(folder=params.folder,name=params.name,cam=params.cam_output[i]) + params.newName.format(cam=params.cam_output[i],time=str(t).zfill(params.Zeros_new)))
if __name__ == "__main__":
    main()