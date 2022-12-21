import numpy as np

def Get_double_thresholding(nms, low_ratio:float, high_ratio:float):
    '''
    input: result of nonmaxsuppression
    low_ratio:  0~1 lwo ratio to max of nms
    high_ratio: 0~1 high ratio to max of nms
    각 텐서별 최대값에 비율을 곱하는 방식으로 사용한다. 
    '''
    double_threshing = np.zeros(nms.shape)
    low_th=np.max(nms)*low_ratio
    high_th=np.max(nms)*high_ratio
    for i in range(0, nms.shape[0]):	
        for j in range(0, nms.shape[1]):
            if nms[i,j] < low_th:	# low보다 작으면 0으로 밀어버림.
                double_threshing[i,j] = 0
            elif nms[i,j] >= low_th and nms[i,j] < high_th: 	# 사이에 있는 것은 중간값으로.
                double_threshing[i,j] = 128
            else:					        
                double_threshing[i,j] = 255  # 큰것은 최대값으로.
    return double_threshing



def Get_hysteresis(double_thresholding:np.ndarray)->np.ndarray:
    '''
    double_thresholding: Result of double thresholding    0 or 128 or 255
    '''
    strong = np.zeros(double_thresholding.shape)
    for i in range(0, double_thresholding.shape[0]-2):		
        for j in range(0, double_thresholding.shape[1]-2):
            val = double_thresholding[i,j]
            if val == 128:			
                #중간값( 위에서 정의함 ) 경우 엣지와 연결되는지 봐야하므로, 경우 나눈다. 
                #본코드에서는 2줄밖까지 확인한다. 픽셀별로 -2,-2 부터 2,2까지 
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