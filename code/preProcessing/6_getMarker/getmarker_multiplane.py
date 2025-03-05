import os, cv2
import numpy as np
import matplotlib.pyplot as plt

from getmarker_functions import *

os.chdir('../../../data/')

# %%

class Target_parameter:
    case_name, Zeros = 'Ilmenau', 2
    cam, t, plane = 0, 1, 1
    alpha = 0.005
    
    minArea , maxArea               = 10, 2000000
    distance_line                   = 10
    kernel                          = (11,11)
    
    N_x, N_y                        = 8, 7
    
    multiplane                      = True

# %%
    
    
def main():
    params = Target_parameter()   
    params.image_origin = params.case_name+"/input/calibration_images/c{cam}/c{cam}_{plane}_{time}.tif"
    params.image_output = params.case_name+"/input/calibration_images/c{cam}/masked/c{cam}_{plane}.tif"
    params.image_output2 = params.case_name+"/input/calibration_images/c{cam}/masked/c{cam}_{plane}_marker.tif"
    params.markerList_output = params.case_name+"/input/calibration_images/c{cam}/marker/c{cam}_{plane}.txt" 
    
    '''
        Masking
    '''

    # define ouput lists
    global mask_points, artifacts, artifacts_add, marker_lines, multiplier
    mask_points, artifacts, artifacts_add, marker_lines, multiplier = [], [], [], [], 1
    img = cv2.imread(params.image_origin.format(cam=params.cam,plane=str(params.plane),time=str(params.t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED)
    img_resize, multiplier = Resize(img,params.alpha)
    img_copy = img.copy()
    
    # create mask , by clicking: leftbot , lefttop, righttop, rightbot
    print('Select mask points: (right-click to finish)')
    mask_points = CollectMask(img_resize,mask_points,multiplier)
    img_masked = Masking(img,np.zeros(img.shape),np.asarray(mask_points))
    print('')
    
    img_masked_2 = img_masked.copy()
    img_masked_2[img_masked_2>np.max(img_masked)/4] = 0
    img_masked_2 = cv2.GaussianBlur(img_masked_2, params.kernel, 0)*100
    img_masked_2[img_masked_2>0] = 50000
    #img[img_masked==0] = np.max(img)
    
    # output marker list
    cv2.imwrite(params.image_output.format(cam=params.cam,plane=str(params.plane)) , img_masked_2)
    
    '''
        Marker detection
    '''
    
    # define ouput lists
    mask_points, artifacts, artifacts_add, marker_lines, multiplier = [], [], [], [], 1
    # load calibration target image
    img = cv2.imread(params.image_output.format(cam=params.cam,plane=str(params.plane)),cv2.IMREAD_UNCHANGED)
    img_o = cv2.imread(params.image_origin.format(cam=params.cam,plane=str(params.plane),time=str(params.t).zfill(params.Zeros)),cv2.IMREAD_UNCHANGED)
    img_resize, multiplier = Resize(img,params.alpha)
    img_copy = img.copy()
    
    img_thresh = cv2.convertScaleAbs(img, alpha=params.alpha)
    
    # find markers on image
    print('Marker search: ')
    cx, cy = RadialSymmetricCenter(img_thresh,params) 
    print(' found ' + str(len(cx)) + ' / ' + str(int(params.N_x*params.N_y)) + '\n')
    
    # marker list sorting, from bot to top, by clicking left right
    print('Search corner marker (left(down,up) -> right(down,up)): (ESC to finish)\n')
    xyl, xyr = CollectMarkerPoints(img_thresh,cx,cy,marker_lines,multiplier)
    centers = np.vstack([cx,cy]).T
    marker_points = FindMarker(xyl,xyr,centers,img_copy,cx,cy,params)
    
    # correct marker grid
    print('Correcting marker grid')
    marker_points_cor = Grid_correction(marker_points,img_copy,params)
    
    # Save or display the result
    plt.figure(figsize=(20,20))
    plt.imshow(img_o,cmap='gray')
    plt.plot(marker_points_cor[:,0],marker_points_cor[:,1],'.')
    plt.savefig(params.image_output2.format(cam=params.cam,plane=str(params.plane)))
    plt.close()
    
    # output marker list
    np.savetxt(params.markerList_output.format(cam=params.cam,plane=str(params.plane)),marker_points_cor,header='x,y')
if __name__ == "__main__":
    main()
