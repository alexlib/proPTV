class Parameter:
    ''' general settings'''
    # input name of case; output name; number of zeros in file names
    case_name, output_name, Zeros  = "LPTC_040", "run1", 5           
    # flag if intial tracks are loaded and path to case where tracks are loaded
    loadOption, load_name, suffix = False, "run1", ""                                                                       
    # cameras + orientation
    cams, depthaxis = [0,1,2,3], [2,2,2,2] # 0=X, 1=Y, 2=Z                    
    # first frame, last frame, initialisation length, delta between frames
    t_start, t_end, t_init, dt = 0, 49, 4, 1
    
    ''' triangulation parameter '''
    # measurment volume [x,y,z]
    Vmin , Vmax = [-600,-600,-600], [600,600,600]   
    # select camera viewing angles 
    startCamForPermute = []
    #  number of triangulation loops, minimum number of different cams needed for triag, distance from epipolar line [px], distance from intersection point of epipolar lines [px], maxmium BackProjection error in Newton Soloff, distance to remove doubled tringulation points                                                          
    N_triag, activeMatches_triag, epsD, epsC, eps, epsDoubling, Imin = 1, 4, 1.0, 0.5, 0.4, 0, [1000,1000,1000,1000]#3, 3, 2, 2, 0.7, 0, 400 
    
    ''' initalisation parameter '''  
    # 2D or 3D intialisation
    modeInit = '3D'
    # maximal absolute tracking velocity for a track                          
    maxvel, angle = 20, 30
    # number of initialisation loops; number of maximal NNs per linking step                                  
    N_init, NN = 4, [3,3,3]
    
    ''' tracking parameter '''
    # active cams for extend                                         
    activeMatches_extend, epsR = 3, 3.0
    # backtracking and gaptracking option
    backtracking, gaptracking = False, False