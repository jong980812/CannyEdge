import numpy as np



def Get_Discrete_angle(Angle):
    '''
    #@ input:   Angle Degree
    #@ output:  Discrete Angle encoding value
    
    Degree ANgle (0, 360) The pixel coordinate have 4 direction
    vertical, horizontal, right-left diagonal
    i divide 360 to 8 part ( 45 * 8 = 360)
    -22.5 ~ 22.5
    22.5  ~ 67.5
    67.5  ~ 112.5
    112.5 ~ 157.5
    ....
    337.5 ~ 360
    THere is 4 pair of 2parts
    1 encoding value for 1 pair (1,2,3,4)
    
    해당 좌표의 그래디언트가 어디를 향하고 있는지 총 4개의 값으로 인코딩.
    각도는 위에 적어놓음.
    
    '''
    discrete_angle = np.zeros_like(Angle)
    angle_w,angle_h=Angle.shape
    for i in range(angle_w):
        for j in range(angle_h):
            if (0 <= Angle[i, j] <= 22.5) or (157.5 <= Angle[i, j] <= 202.5) or (337.5 < Angle[i, j] < 360):
                discrete_angle[i, j] = 1
            elif (22.5 <= Angle[i, j] <= 67.5) or (202.5 <= Angle[i, j] <= 247.5):
                discrete_angle[i, j] = 2
            elif (67.5 <= Angle[i, j] <= 112.5) or (247.5 <= Angle[i, j] <= 292.5):
                discrete_angle[i, j] = 3
            elif (112.5 <= Angle[i, j] <= 157.5) or (292.5 <= Angle[i, j] <= 337.5):
                discrete_angle[i, j] = 4
                # 해당하는 각도와 값 매핑.
    return discrete_angle

def Get_Nms(discrete_angle, magnitude):
    nms = np.zeros(discrete_angle.shape) #@ Empty nms
    w, h = nms.shape
    # 마지막 칸은 비교할 필요 없다.
    #! 각 각도와 수직한 방향의 앞 , 뒤 픽셀을 비교하기 위한 코드. 각 인코딩값과 매칭시킴.
    for i in range(w-1):
        for j in range(h-1):
            if discrete_angle[i,j] == 1:
                neighbor_a,neighbor_b=magnitude[i,j-1], magnitude[i,j+1]
            elif discrete_angle[i,j]==2:
                neighbor_a,neighbor_b=magnitude[i-1,j+1], magnitude[i+1,j-1]
            elif discrete_angle[i,j] == 3:
                neighbor_a,neighbor_b=magnitude[i-1,j], magnitude[i+1,j]
            elif discrete_angle[i,j] == 4:
                neighbor_a,neighbor_b=magnitude[i-1,j-1], magnitude[i+1,j+1]
            #! 각 인코딩 별로 앞뒤 픽셀을 정한다.
            
            if magnitude[i,j] >= neighbor_a and magnitude[i,j] >=neighbor_b:
                nms[i,j]=magnitude[i,j]     
            #! 이웃 픽셀과 비교해서 자신이 최대가 아니면 0으로 바꾼다. 
    return nms
