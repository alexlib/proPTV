'''

    This script contains functions for image processing

'''


import cv2, sys
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from scipy.signal import convolve2d


def ImageProcessing(cam, t, i, times, params):
    # load image
    img_origin = cv2.imread(params.image_input.format(cam=cam,time=str(t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED)
    # create processing img
    img = img_origin.copy()
    
    # averaging           
    a = i-(params.window//2) if i>=(params.window//2) else 0 
    a = len(times)-params.window if i>=(len(times)-(params.window//2)) else a 
    b = i+(params.window//2) if i<(len(times)-(params.window//2)) else len(times)-1
    b = params.window-1 if i<(params.window//2) else b
    min_img = params.weight_min*np.min([cv2.imread(params.image_input.format(cam=cam,time=str(ti).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED) for ti in times[a:b+1]], axis=0)
    img = img-min_img
    
    # thresholding
    img[min_img>params.threshold_minimg] = 0
    img[img<params.threshold] = 0
    
    # masking
    mask = cv2.imread(params.mask_path.format(cam=cam),cv2.IMREAD_UNCHANGED)
    img[mask==0] = 0
    
    # short out to check parameter
    if params.debug == True:
        if params.show == True:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,8), sharex=True, sharey=True)
            ax1.imshow(img_origin,cmap='gray',vmax=np.mean(img_origin[img>params.threshold]))
            ax2.imshow(min_img,cmap='gray', vmax=np.mean(img_origin[img>params.threshold]))
            ax3.imshow(img,cmap='gray',vmax=np.mean(img[img>0]))
            plt.tight_layout(), plt.show()
            sys.exit()
    
    # delete single artifacts
    if params.delete_artifacts == True:
        img[convolve2d(img, [[1,1,1],[1,0,1],[1,1,1]], mode='same')==0] = 0
        
    # blur
    if params.blur == True:
        img = cv2.GaussianBlur(img, params.Gauskernel, params.Gauskernelstd)
    img = np.rint(img) 

    # create particle list for peak detection
    finalList = []
    img_peak = img.copy().astype('float')
    height, width = img_peak.shape[:2]
    y_coords, x_coords = np.mgrid[:height, :width]
       
    # peak search
    for n in range(params.runs):
        # peak detection
        particleList = peak_local_max(img_peak, min_distance=int(params.particleSize),num_peaks=params.maxParticle-len(finalList))
        CX , CY = [] , []
        for x,y in zip(particleList[:,1],particleList[:,0]):
            binsX, binsY = np.array([x,x+1,x-1],dtype=int), np.array([y,y+1,y-1],dtype=int)
            valueX , valueY = img_origin[y,binsX] , img_origin[binsY,x]
            meanX, meanY = np.sum(binsX*valueX) / np.sum(valueX), np.sum(binsY*valueY) / np.sum(valueY)
            CX.append(meanX) , CY.append(meanY)
            # remove peak
            IDx0, IDx1 = int(np.rint(meanX))-params.particleSize, int(np.rint(meanX))+(params.particleSize+1)
            IDy0, IDy1 = int(np.rint(meanY))-params.particleSize, int(np.rint(meanY))+(params.particleSize+1)
            G = img_peak[int(np.rint(meanY)),int(np.rint(meanX))] * np.exp( -0.5*(((x_coords[IDy0:IDy1,IDx0:IDx1]-meanX)/params.std)**2+((y_coords[IDy0:IDy1,IDx0:IDx1]-meanY)/params.std)**2) )
            img_peak[IDy0:IDy1,IDx0:IDx1] -= G 
        img_peak[img_peak<params.threshold] = 0
        particleList = np.vstack([CX,CY]).T
        if len(particleList)>0:
            if params.debug == True:
                print('run '+ str(n) + ' - found ' +str(len(particleList)) + ' particle centers')
            finalList += list(particleList)
    finalList = np.asarray(finalList)
    finalList, IDs_unique = np.unique(finalList,axis=0,return_index=True)
        
    # output a processed image
    img_proc = np.zeros_like(img).astype('uint8')
    img_proc[np.asarray(np.round(finalList[:,1]),dtype=int),np.asarray(np.round(finalList[:,0]),dtype=int)] = 250
    img_proc = cv2.GaussianBlur( img_proc , [3,3] , 1 )*3
    
    # save proc image and particle list
    if params.debug == False:
        cv2.imwrite(params.image_output.format(cam=cam,time=str(t).zfill(params.Zeros)) , img_proc)
        np.savetxt( params.particleList_output.format(cam=cam,time=str(t).zfill(params.Zeros)) , finalList , header="center_x , center_y")
    return finalList, img_origin, img, img_proc, min_img
