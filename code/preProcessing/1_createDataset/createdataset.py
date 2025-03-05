'''
    Enter the parameter and run the script.
    The data structure of the dataset is build. 
'''

import os 

# %%

class Dataset_parameter:
    # name of the dataset
    NameOfDataSet = 'LPTC_01'
    # number of cameras observing the flow
    NumberOfCams = 4

# %%

def main():
    params = Dataset_parameter()
    
    # if the path does not exist, create dataset structure
    path = '../../../data/' + params.NameOfDataSet
    if not os.path.exists(path):
        os.mkdir( path )
        os.mkdir( path + "/input" ) 
        os.mkdir( path + "/input/calibration_images" )
        [os.mkdir( path + "/input/calibration_images/c"+str(i) ) for i in range(params.NumberOfCams)]
        [os.mkdir( path + "/input/calibration_images/c"+str(i)+"/masked" ) for i in range(params.NumberOfCams)]
        [os.mkdir( path + "/input/calibration_images/c"+str(i)+"/marker" ) for i in range(params.NumberOfCams)]
        os.mkdir( path + "/input/calibration" )
        [os.mkdir( path + "/input/calibration/c"+str(i) ) for i in range(params.NumberOfCams)]
        os.mkdir( path + "/input/masks" )
        [os.mkdir( path + "/input/masks/c"+str(i) ) for i in range(params.NumberOfCams)]
        os.mkdir( path + "/input/raw_images" )
        [os.mkdir( path + "/input/raw_images/c"+str(i) ) for i in range(params.NumberOfCams)]
        os.mkdir( path + "/input/processed_images" )
        [os.mkdir( path + "/input/processed_images/c"+str(i) ) for i in range(params.NumberOfCams)]
        os.mkdir( path + "/input/particle_lists" )
        [os.mkdir( path + "/input/particle_lists/c"+str(i) ) for i in range(params.NumberOfCams)]
        os.mkdir( path + "/output" )
        os.mkdir( path + "/output/VSC" )
        os.mkdir( path + "/analysis" )
if __name__ == "__main__":
    main()