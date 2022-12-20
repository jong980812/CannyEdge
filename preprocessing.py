import numpy as np

#TODO Converting Image to Gray scale  1 channel
def Convert_gray(image : np.ndarray)->np.ndarray:
    '''
    #@ Ref: https://stackoverflow.com/questions/61058335/open-cv-how-does-the-bgr2gray-function-work
    Conver image to gray 
    Same cv2.cvtColor
    '''
    gray=np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray[i][j]+=0.299*image[i][j][2]+0.587*image[i][j][1] + 0.114*image[i][j][0]
    assert gray.shape == (image.shape[0],image.shape[1]), f"Error -> converting Failed : two ndarray's shape is different"
    
    return gray
