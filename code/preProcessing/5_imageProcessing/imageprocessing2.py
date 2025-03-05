'''

    This script does the image processing and creates the particle list.
    
'''

import joblib, os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from imageprocessing2_functions import *

os.chdir('../../../data/')

# %%

class Imgproc_parameter:
    '''
    case_name, Zeros = 'LPTC_040', 5
    t_start , t_end, cpu = 0, 49, 25#joblib.cpu_count()
    cams = [1,2,3]
    
    debug, show  = False, False
    
    window, weight_min = 3, 1.2 # 0, 0 | 3, 1.2
    threshold, threshold_minimg = 50, 200000000 # 0,0 | 18, 2000
    
    delete_artifacts = False
    blur, Gauskernel, Gauskernelstd = False, [3,3], 1.0
    
    runs, std = 7, 0.7 #5, 1.0
    maxParticle, particleSize = 550000, 2
    '''
    case_name, Zeros = 'Ilmenau', 4
    t_start , t_end, cpu = 50000, 50200, 25#joblib.cpu_count()
    cams = [0,1,2,3]
    
    debug, show  = False, False
    
    window, weight_min = 3, 1.1 # 0, 0 | 3, 1.2
    threshold, threshold_minimg = 500, 200000000 # 0,0 | 18, 2000
    
    delete_artifacts = False
    blur, Gauskernel, Gauskernelstd = False, [3,3], 1.0
    
    runs, std = 1, 0.7 #5, 1.0
    maxParticle, particleSize = 20000, 2
    
# %%

def main():
    # load parameter
    params = Imgproc_parameter()
    params.mask_path = params.case_name + "/input/masks/c{cam}/c{cam}_mask.tif"
    params.image_input = params.case_name + "/input/raw_images/c{cam}/c{cam}_{time}.tif"
    params.image_output = params.case_name + "/input/processed_images/c{cam}/c{cam}_{time}_proc.tif"
    params.particleList_output = params.case_name + "/input/particle_lists/c{cam}/c{cam}_{time}.txt"
    
    # check for debug mode and run image processing
    times = np.linspace(params.t_start,params.t_end,params.t_end-params.t_start+1,dtype=int)
    if params.debug == True:
        p, img_origin, img, img_proc, min_img = ImageProcessing(params.cams[0], params.t_start, 0, times, params)
        print(' found: ', len(p), ' particles')
        # plot result
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,6), sharex=True, sharey=True)
        ax1.imshow(img_origin,cmap='gray',vmax=np.mean(img_origin[img_origin>params.threshold])), ax1.plot(p[:,0],p[:,1],'.',c='red')
        ax2.imshow(min_img,cmap='gray', vmax=np.mean(img_origin[img_origin>params.threshold]))
        ax3.imshow(img,cmap='gray',vmax=np.mean(img[img>0])), ax3.plot(p[:,0],p[:,1],'.',c='red')
        ax4.imshow(img_proc,cmap='gray',vmax=85)
        plt.tight_layout(), plt.show()
    else:
        for c in params.cams:
            print('Image processing camera ' + str(c))
            joblib.Parallel(n_jobs=params.cpu)(joblib.delayed(ImageProcessing)(c,t,i,times,params) for i,t in enumerate(tqdm(times ,desc=' processing', position=0, leave=True, delay=0.5)))
if __name__ == "__main__":
    main()
