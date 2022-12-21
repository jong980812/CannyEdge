import argparse
import os
import cv2
import utils
from preprocessing import Convert_gray, Get_filter_size,Get_filter_scope
from gradient import Get_gradient, Gradient_Magnitude_Angle,Get_edge_LoG
from smoothing import Smoothing
from nms import Get_Discrete_angle, Get_Nms
from thresholding import Get_double_thresholding,Get_hysteresis
from utils import print_bar as bar


'''
Keyword

filter_scope: 설정해놓은 공식대로 원하는 가우시안 threshold까지 해당 가우시안이 몇 칸 가야하는지 알려준다.
filter_size: 가우시안이 가야할 칸 * 2 를 한 뒤 홀수가 되야하므로 1개 더해서 이것이 필터의 사이즈가 된다. 


'''



def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--image_path',default=None,type=str,                               help='이미지 경로')
    parser.add_argument('--output_img_name', type=str, default='result',                    help="저장할 이미지의 고유 이름")
    parser.add_argument('--run_dir', default='./', type=str,                                help='현재 돌리는 파일 디렉토리' )
    parser.add_argument('--smoothing', default=None, choices=['gaussian', 'box'], type=str, help='스무딩 방식 선택')
    parser.add_argument('--sigma', default=1.0, type=float,                                 help='표준편차')
    parser.add_argument('--th', default=0.01, type=float,                                   help='가우시안 사용할 값 임계치')
    parser.add_argument('--filter_size', default=None, type=int,                            help='Direct setting filter size, This is needed for LoG')
    parser.add_argument('--gradient_method', default='sobel', 
                       choices=['Sobel','Prewitt','LoG','DoG','Scharr'],                    help='미분 필터 선택.')   
    parser.add_argument('--log_th', type=float,default=0.5,                                 help='LoG를 이용하여 구한 magnitude의 th값.')
    parser.add_argument('--hysterisis_th',nargs='+',type=float,default=[0.1,0.2],           help="double thresholding할 때 값. 0~1이어야함.")

    return parser.parse_args()

def main(args):
    img = cv2.imread(args.image_path) # 이미지 읽기.
    gray_img=Convert_gray(img) # 흑백으로 변환한다.
    print(f"Method is {args.gradient_method}") 
    output_path=os.path.join(args.run_dir,'result',args.output_img_name)
    bar()
    #@ LoG의 경우 따로 정의.
    if args.gradient_method == "LoG":
        assert args.filter_size==5 or args.filter_size==9 #! 필터 사이즈 2개 밖에 없음. 
        Log_mag,_=Gradient_Magnitude_Angle(gray_img, args.gradient_method, args.filter_size)
        utils.Save_img(Log_mag,output_path,'LoG_magnitude')
        Log_edge=Get_edge_LoG(Log_mag,args.log_th)
        utils.Save_img(Log_edge,output_path,'LoG_edge')
        return 
    
    # 필터 사이즈가 따로 지정되지 않았으면 공식으로 추정해줌.
    if args.filter_size is None:
        filter_scope = int(Get_filter_scope(args.th, args.sigma))
        filter_size = int(Get_filter_size(filter_scope))
    else:
    #필터 사이즈가 있으면, 정해주고, scope는 역으로 계산.
        filter_size=args.filter_size
        filter_scope=int((filter_size-1)/2)
    print("\n\n\n-------------START--------------------------------")
    
    #스무딩 후 이미지 저장.
    print("Smoothing");bar()
    smoothing_img=Smoothing(gray_img, args.smoothing, args.sigma,filter_scope, filter_size)
    print("...Complete!");bar();print()
    utils.Save_img(smoothing_img,output_path,'Smoothing_img')
    
    #그래디언트 크기와 각도 뽑고, 저장.
    print("Calculating Gradient");bar()
    G_mag,G_Angle=Gradient_Magnitude_Angle(gray_img, args.gradient_method, filter_size, args.sigma)
    utils.Save_img(G_mag,output_path,"Magnitude")
    utils.Save_img(G_Angle,output_path,"Angle")
    print("...Complete!");bar();print()
    
    #비최대 억제
    print("Nonmax Surpression");bar()    
    discrete_angle=Get_Discrete_angle(G_Angle)
    nms=Get_Nms(discrete_angle, G_mag)
    utils.Save_img(nms,output_path,"Nms")
    print("...Complete!");bar();print()
    
    #이중 스레시홀딩.
    print("Double Thresholding");bar()
    double=Get_double_thresholding(nms,args.hysterisis_th[0],args.hysterisis_th[1])
    utils.Save_img(double,output_path,"Double_th")
    print("...Complete!");bar();print()
    
    print("Hysterisis Thresholding");bar()
    hys=Get_hysteresis(double)
    utils.Save_img(hys,output_path,"Hysterisis")
    print("...Complete!");bar()
    #TODO 커널과 필터 분명히 하기.
    
    
if __name__== "__main__":
    args=get_args()
    if not os.path.isdir(os.path.join(args.run_dir,'result')):  
        #@ Check output Diretory.
        '''
        If there is no 'result' directory in your run directory, 
        Make the result folder.
        '''
        output_path=os.path.join(args.run_dir,'result')
        os.mkdir(output_path)
    utils.arg_print(args)
    main(args)