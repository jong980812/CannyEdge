import numpy as np

def Get_double_thresholding(nms, low_ratio:float, high_ratio:float):
    '''
    input: result of nonmaxsuppression
    low_ratio:  0~1 lwo ratio to max of nms
    high_ratio: 0~1 high ratio to max of nms
    '''
    double_threshing = np.zeros(nms.shape)
    low_th=np.max(nms)*low_ratio
    high_th=np.max(nms)*high_ratio
    for i in range(0, nms.shape[0]):	
        for j in range(0, nms.shape[1]):
            if nms[i,j] < low_th:	# lower than low threshold
                double_threshing[i,j] = 0
            elif nms[i,j] >= low_th and nms[i,j] < high_th: 	# between thresholds
                double_threshing[i,j] = 128
            else:					        # higher than high threshold
                double_threshing[i,j] = 255
    return double_threshing



def Get_hysteresis(double_thresholding:np.ndarray)->np.ndarray:
    '''
    double_thresholding: Result of double thresholding    0 or 128 or 255
    '''
    strong = np.zeros(double_thresholding.shape)
    for i in range(0, double_thresholding.shape[0]-2):		
        for j in range(0, double_thresholding.shape[1]-2):
            val = double_thresholding[i,j]
            if val == 128:			# check if weak edge connected to strong
                for a in range(-2,3):
                    for b in range(-2,3):
                        if double_thresholding[i-a,j-b] == 255 : 
                            strong[i,j] = 255
                # if double_thresholding[i-1,j] == 255 or \
                #     double_thresholding[i+1,j] == 255 or \
                #     double_thresholding[i-1,j-1] == 255 or \
                #     double_thresholding[i+1,j-1] == 255 or \
                #     double_thresholding[i-1,j+1] == 255 or \
                #     double_thresholding[i+1,j+1] == 255 or \
                #     double_thresholding[i,j-1] == 255 or \
                #     double_thresholding[i,j+1] == 255 :
                #  g_strong[i,j] = 255		# replace weak edge as strong
            elif val == 255:
                strong[i,j] = 255		# strong edge remains as strong edge
    return strong